"""End-to-end market ingestion and validation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout

from ..config import AresConfig
from ..exceptions import AresError, DataQualityError
from ..utils import atomic_write_json, timeframe_to_seconds, utc_now
from .providers import CCXTOHLCVProvider, OHLCVProvider
from .quality import QualityReport, validate_cross_venue, validate_ohlcv
from .storage import (
    market_path,
    merge_market,
    quality_path,
    read_market,
    sync_duckdb,
    write_market,
)


@dataclass(slots=True)
class ExchangeIngestion:
    exchange: str
    path: Path
    rows: int
    new_rows: int
    report: QualityReport


@dataclass(slots=True)
class IngestionResult:
    exchanges: list[ExchangeIngestion]
    cross_venue_reports: list[QualityReport]
    passed: bool
    committed: bool
    generated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "committed": self.committed,
            "generated_at": self.generated_at,
            "exchanges": [
                {
                    "exchange": item.exchange,
                    "path": item.path,
                    "rows": item.rows,
                    "new_rows": item.new_rows,
                    "report": item.report.to_dict(),
                }
                for item in self.exchanges
            ],
            "cross_venue_reports": [report.to_dict() for report in self.cross_venue_reports],
        }


def _closed_candles_only(frame: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    now = now or utc_now()
    seconds = timeframe_to_seconds(timeframe)
    current_start_seconds = int(now.astimezone(UTC).timestamp()) // seconds * seconds
    current_start = pd.Timestamp(current_start_seconds, unit="s", tz="UTC")
    return frame.loc[pd.to_datetime(frame["timestamp"], utc=True) < current_start].copy()


def _resume_since(existing: pd.DataFrame, configured_since: datetime, timeframe: str) -> datetime:
    if existing.empty:
        return configured_since
    step = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
    last = pd.to_datetime(existing["timestamp"], utc=True).max()
    return max(configured_since, (last + step).to_pydatetime())


def ingest_market_data(
    config: AresConfig,
    *,
    providers: dict[str, OHLCVProvider] | None = None,
) -> IngestionResult:
    """Serialize ingestion so concurrent jobs cannot overwrite each other's snapshots."""
    config.storage.root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(config.storage.root / ".ares-ingest.lock", timeout=0)
    try:
        with lock:
            return _ingest_market_data_unlocked(config, providers=providers)
    except Timeout as exc:
        raise AresError("Another ARES ingestion is already running") from exc


def _ingest_market_data_unlocked(
    config: AresConfig,
    *,
    providers: dict[str, OHLCVProvider] | None = None,
) -> IngestionResult:
    """Fetch primary and validation venues, persist Parquet, and enforce quality gates."""
    exchanges = [config.data.primary_exchange, *config.data.validation_exchanges]
    exchange_results: list[ExchangeIngestion] = []
    frames: dict[str, pd.DataFrame] = {}
    generated_at = utc_now()

    for exchange in exchanges:
        path = market_path(config.storage.root, exchange, config.data.symbol, config.data.timeframe)
        existing = read_market(path)
        since = (
            _resume_since(existing, config.data.since, config.data.timeframe)
            if config.data.resume
            else config.data.since
        )
        provider = (providers or {}).get(exchange) or CCXTOHLCVProvider(exchange)
        incoming = provider.fetch_range(
            config.data.symbol,
            config.data.timeframe,
            since,
            config.data.until,
            limit=config.data.page_limit,
            max_pages=config.data.max_pages,
            retries=config.data.retries,
        )
        if config.data.drop_open_candle:
            incoming = _closed_candles_only(incoming, config.data.timeframe)
        merged = merge_market(existing, incoming)
        report = validate_ohlcv(
            merged,
            config.data.timeframe,
            source=exchange,
            max_gap_count=config.data.max_gap_count,
            expected_exchange=exchange,
            expected_symbol=config.data.symbol,
            expected_timeframe=config.data.timeframe,
            as_of=generated_at if config.data.until is None else None,
            max_staleness_bars=(
                config.data.max_staleness_bars if config.data.until is None else None
            ),
            expected_start=config.data.since,
            expected_end=config.data.until,
        )
        frames[exchange] = merged
        exchange_results.append(
            ExchangeIngestion(
                exchange=exchange,
                path=path,
                rows=len(merged),
                new_rows=len(incoming),
                report=report,
            )
        )

    primary = frames[config.data.primary_exchange]
    cross_reports: list[QualityReport] = []
    required_cross_columns = {"timestamp", "close"}
    for exchange in config.data.validation_exchanges:
        if not required_cross_columns.issubset(primary.columns) or not required_cross_columns.issubset(
            frames[exchange].columns
        ):
            continue
        report = validate_cross_venue(
            primary,
            frames[exchange],
            primary_name=config.data.primary_exchange,
            secondary_name=exchange,
            max_p95_bps=config.data.max_cross_venue_p95_bps,
            min_overlap=config.data.min_cross_venue_overlap,
            max_latest_bps=config.data.max_cross_venue_latest_bps,
        )
        cross_reports.append(report)
    passed = all(item.report.passed for item in exchange_results) and all(
        report.passed for report in cross_reports
    )
    result = IngestionResult(
        exchanges=exchange_results,
        cross_venue_reports=cross_reports,
        passed=passed,
        committed=False,
        generated_at=generated_at,
    )
    for item in exchange_results:
        atomic_write_json(quality_path(config.storage.root, item.exchange), item.report.to_dict())
    for report in cross_reports:
        atomic_write_json(quality_path(config.storage.root, report.source), report.to_dict())
    atomic_write_json(config.storage.root / "quality" / "latest.json", result.to_dict())
    if not passed:
        failures = [
            issue.message
            for item in exchange_results
            for issue in item.report.issues
        ] + [issue.message for report in cross_reports for issue in report.issues]
        if config.data.fail_on_quality:
            raise DataQualityError("Market data failed quality gates: " + "; ".join(failures))
        return result

    # Validate every venue and every cross-check before replacing any canonical file. Write the primary
    # last so a partial filesystem failure cannot make the main feed look newer than its validators.
    commit_order = [*config.data.validation_exchanges, config.data.primary_exchange]
    paths = {item.exchange: item.path for item in exchange_results}
    for exchange in commit_order:
        write_market(paths[exchange], frames[exchange])
    sync_duckdb(config.storage.duckdb_path, config.storage.root)
    result.committed = True
    atomic_write_json(config.storage.root / "quality" / "latest.json", result.to_dict())
    return result
