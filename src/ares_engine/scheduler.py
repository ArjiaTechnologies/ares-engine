"""APScheduler quick/deep cycles for ingestion, search, export, and promotion."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout

from .config import AresConfig, load_config
from .data.ingest import IngestionResult, ingest_market_data
from .data.storage import market_path, read_market
from .exceptions import DataQualityError
from .live import generate_paper_signal
from .promotion import promote, resolve_champion
from .search import run_search
from .training import train_candidate

LOGGER = logging.getLogger(__name__)


@contextmanager
def _cycle_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(root / ".ares-cycle.lock", timeout=0)
    try:
        with lock:
            yield
    except Timeout as exc:
        raise RuntimeError("Another ARES quick/deep cycle is already running") from exc


def _require_committed_ingestion(result: IngestionResult) -> None:
    if result.passed and result.committed:
        return
    failures = [
        issue.message
        for exchange in result.exchanges
        for issue in exchange.report.issues
    ] + [
        issue.message
        for report in result.cross_venue_reports
        for issue in report.issues
    ]
    detail = "; ".join(failures) if failures else "ingestion did not commit canonical data"
    raise DataQualityError(f"Scheduler cycle stopped after failed ingestion: {detail}")


def _read_validation_frames(config: AresConfig) -> dict[str, pd.DataFrame]:
    return {
        exchange: read_market(
            market_path(
                config.storage.root,
                exchange,
                config.data.symbol,
                config.data.timeframe,
            )
        )
        for exchange in config.data.validation_exchanges
    }


def quick_cycle(config_path: Path) -> None:
    config = load_config(config_path)
    with _cycle_lock(config.storage.root):
        ingestion = ingest_market_data(config)
        _require_committed_ingestion(ingestion)
        champion = resolve_champion(config.storage.artifacts)
        if champion is None:
            LOGGER.warning("No champion exists; quick cycle ends after ingestion")
            return
        primary = read_market(
            market_path(
                config.storage.root,
                config.data.primary_exchange,
                config.data.symbol,
                config.data.timeframe,
            )
        )
        generate_paper_signal(
            champion,
            primary,
            secondary_ohlcv=_read_validation_frames(config),
            log_path=config.storage.root / "paper" / "signals.jsonl",
        )


def deep_cycle(config_path: Path) -> None:
    config = load_config(config_path)
    with _cycle_lock(config.storage.root):
        ingestion = ingest_market_data(config)
        _require_committed_ingestion(ingestion)
        source = market_path(
            config.storage.root,
            config.data.primary_exchange,
            config.data.symbol,
            config.data.timeframe,
        )
        frame = read_market(source)
        candidate_config = config
        if config.scheduler.run_search_on_deep:
            _, candidate_config = run_search(frame, config)
        bundle, _ = train_candidate(
            frame,
            candidate_config,
            source_path=source,
            repository_root=Path.cwd(),
        )
        promote(bundle, candidate_config.storage.artifacts, candidate_config.gates)


def run_scheduler(config_path: Path) -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError as exc:
        raise RuntimeError("APScheduler is not installed") from exc
    config = load_config(config_path)
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        quick_cycle,
        "interval",
        minutes=config.scheduler.quick_minutes,
        args=[config_path],
        id="ares-quick",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        deep_cycle,
        "interval",
        hours=config.scheduler.deep_hours,
        args=[config_path],
        id="ares-deep",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
