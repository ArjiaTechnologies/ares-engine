from datetime import datetime
from pathlib import Path

import pandas as pd

from ares_engine.config import load_config
from ares_engine.data.ingest import ingest_market_data
from ares_engine.synthetic import make_synthetic_ohlcv


class FakeProvider:
    def __init__(self, exchange_id: str, frame: pd.DataFrame) -> None:
        self.exchange_id = exchange_id
        self.frame = frame

    def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None,
        *,
        limit: int,
        max_pages: int,
        retries: int,
    ) -> pd.DataFrame:
        del symbol, timeframe, since, until, limit, max_pages, retries
        return self.frame.copy()


def _config(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.duckdb_path = tmp_path / "data" / "ares.duckdb"
    config.data.since = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    config.data.drop_open_candle = False
    config.data.fail_on_quality = False
    config.data.max_gap_count = 0
    return config


def test_failed_cross_venue_gate_does_not_replace_canonical_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    primary = make_synthetic_ohlcv(200, seed=40, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    secondary[["open", "high", "low", "close"]] *= 1.05
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    config.data.max_cross_venue_p95_bps = 50.0
    writes: list[str] = []

    monkeypatch.setattr("ares_engine.data.ingest.read_market", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        "ares_engine.data.ingest.write_market",
        lambda path, frame: writes.append(path.as_posix()),
    )
    monkeypatch.setattr(
        "ares_engine.data.ingest.sync_duckdb", lambda *args: writes.append("duckdb")
    )

    result = ingest_market_data(
        config,
        providers={
            "coinbase": FakeProvider("coinbase", primary),
            "kraken": FakeProvider("kraken", secondary),
        },
    )

    assert not result.passed
    assert not result.committed
    assert writes == []
    assert any(not report.passed for report in result.cross_venue_reports)


def test_current_ingestion_rejects_stale_venues_before_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.data.until = None
    primary = make_synthetic_ohlcv(200, seed=41, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    reference = primary["timestamp"].max() + pd.Timedelta(hours=10)
    writes: list[str] = []

    monkeypatch.setattr("ares_engine.data.ingest.utc_now", lambda: reference.to_pydatetime())
    monkeypatch.setattr("ares_engine.data.ingest.read_market", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        "ares_engine.data.ingest.write_market",
        lambda path, frame: writes.append(path.as_posix()),
    )
    monkeypatch.setattr(
        "ares_engine.data.ingest.sync_duckdb", lambda *args: writes.append("duckdb")
    )

    result = ingest_market_data(
        config,
        providers={
            "coinbase": FakeProvider("coinbase", primary),
            "kraken": FakeProvider("kraken", secondary),
        },
    )

    assert not result.passed
    assert not result.committed
    assert writes == []
    assert all(
        any(issue.code == "stale_data" for issue in exchange.report.issues)
        for exchange in result.exchanges
    )


def test_ingestion_lock_fails_closed_when_another_ingestion_is_running(tmp_path: Path) -> None:
    import pytest
    from filelock import FileLock

    from ares_engine.exceptions import AresError

    config = _config(tmp_path)
    config.storage.root.mkdir(parents=True)
    with FileLock(config.storage.root / ".ares-ingest.lock"):
        with pytest.raises(AresError, match="already running"):
            ingest_market_data(config, providers={})


def test_truncated_requested_range_does_not_commit(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = make_synthetic_ohlcv(200, seed=42, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=3)).to_pydatetime()
    writes: list[str] = []

    monkeypatch.setattr("ares_engine.data.ingest.read_market", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        "ares_engine.data.ingest.write_market",
        lambda path, frame: writes.append(path.as_posix()),
    )
    monkeypatch.setattr(
        "ares_engine.data.ingest.sync_duckdb", lambda *args: writes.append("duckdb")
    )

    result = ingest_market_data(
        config,
        providers={
            "coinbase": FakeProvider("coinbase", primary),
            "kraken": FakeProvider("kraken", secondary),
        },
    )

    assert not result.passed
    assert not result.committed
    assert writes == []
    assert all(
        any(issue.code == "incomplete_end_coverage" for issue in exchange.report.issues)
        for exchange in result.exchanges
    )
