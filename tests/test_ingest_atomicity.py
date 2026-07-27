"""Crash, concurrency, and canonical-state integrity tests for ingestion storage."""

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb
import pandas as pd
import pytest

import ares_engine.data.ingest as ingest_module
from ares_engine.config import load_config
from ares_engine.data.ingest import ingest_market_data, recover_pending_commit
from ares_engine.data.storage import (
    TEMP_SUFFIX,
    market_path,
    read_market,
    sync_duckdb,
    write_market,
)
from ares_engine.exceptions import AresError
from ares_engine.synthetic import make_synthetic_ohlcv


class FakeProvider:
    def __init__(self, exchange_id: str, frame: pd.DataFrame) -> None:
        self.exchange_id = exchange_id
        self.frame = frame

    def fetch_range(self, symbol, timeframe, since, until, *, limit, max_pages, retries):
        output = self.frame[self.frame["timestamp"] >= pd.Timestamp(since)].copy()
        if until is not None:
            output = output[output["timestamp"] < pd.Timestamp(until)]
        return output


def _config(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.artifacts = tmp_path / "artifacts"
    config.storage.duckdb_path = tmp_path / "data" / "ares.duckdb"
    config.data.since = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    config.data.drop_open_candle = False
    config.data.fail_on_quality = True
    config.data.max_gap_count = 0
    config.data.max_cross_venue_p95_bps = 20.0
    config.data.min_cross_venue_overlap = 50
    return config


def _providers(primary: pd.DataFrame):
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    secondary["close"] *= 1.0001
    secondary["high"] = secondary[["high", "close"]].max(axis=1)
    return {
        "coinbase": FakeProvider("coinbase", primary),
        "kraken": FakeProvider("kraken", secondary),
    }


def test_crashed_temp_file_cannot_poison_duckdb_view(tmp_path: Path) -> None:
    """Regression for the audit-reproduced defect: stale staging files were globbed."""
    root = tmp_path / "data"
    raw = root / "raw" / "coinbase" / "eth-usd"
    raw.mkdir(parents=True)
    good = make_synthetic_ohlcv(5, seed=1, exchange="coinbase")
    write_market(raw / "1h.parquet", good)
    # Both an old-style crashed temp (canonical suffix) and a new-style one:
    poisoned = good.copy()
    poisoned["close"] *= 100
    poisoned.to_parquet(raw / ".1h.crashed.parquet", index=False)
    poisoned.to_parquet(raw / f".1h.crashed{TEMP_SUFFIX}", index=False)
    sync_duckdb(root / "ares.duckdb", root)
    with duckdb.connect(str(root / "ares.duckdb")) as connection:
        rows = connection.execute("SELECT count(*) FROM ohlcv").fetchone()[0]
        files = [
            Path(f[0]).name
            for f in connection.execute("SELECT DISTINCT filename FROM ohlcv").fetchall()
        ]
    assert rows == 5, "hidden/staging files must never appear in the canonical view"
    assert files == ["1h.parquet"]


def test_interrupted_commit_is_rolled_forward_before_next_ingestion(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    primary = make_synthetic_ohlcv(200, seed=11, exchange="coinbase")
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    providers = _providers(primary)

    real_commit = ingest_module.commit_staged
    calls: list[str] = []

    def crashing_commit(temp_path, final_path):
        calls.append(final_path.as_posix())
        if len(calls) == 1:
            real_commit(temp_path, final_path)
            raise OSError("simulated crash between the first and second rename")
        real_commit(temp_path, final_path)

    monkeypatch.setattr(ingest_module, "commit_staged", crashing_commit)
    with pytest.raises(OSError, match="simulated crash"):
        ingest_market_data(config, providers=providers)

    # State right now is torn: kraken renamed, coinbase not, journal still present.
    journal = config.storage.root / ".ares-commit-journal.json"
    assert journal.exists()
    assert market_path(config.storage.root, "kraken", "ETH/USD", "1h").exists()
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()

    monkeypatch.setattr(ingest_module, "commit_staged", real_commit)
    recovered = recover_pending_commit(config)
    assert recovered is True
    assert not journal.exists()
    coinbase = read_market(market_path(config.storage.root, "coinbase", "ETH/USD", "1h"))
    kraken = read_market(market_path(config.storage.root, "kraken", "ETH/USD", "1h"))
    assert len(coinbase) == 200 and len(kraken) == 200
    assert not list((config.storage.root / "raw").rglob(f"*{TEMP_SUFFIX}"))
    with duckdb.connect(str(config.storage.duckdb_path)) as connection:
        assert connection.execute("SELECT count(*) FROM ohlcv").fetchone()[0] == 400


def test_recovery_runs_automatically_at_next_locked_ingestion(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    primary = make_synthetic_ohlcv(150, seed=12, exchange="coinbase")
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    providers = _providers(primary)

    real_commit = ingest_module.commit_staged
    state = {"count": 0}

    def crash_after_first(temp_path, final_path):
        state["count"] += 1
        real_commit(temp_path, final_path)
        if state["count"] == 1:
            raise OSError("crash")

    monkeypatch.setattr(ingest_module, "commit_staged", crash_after_first)
    with pytest.raises(OSError):
        ingest_market_data(config, providers=providers)
    monkeypatch.setattr(ingest_module, "commit_staged", real_commit)

    result = ingest_market_data(config, providers=providers)
    assert result.passed and result.committed
    assert not (config.storage.root / ".ares-commit-journal.json").exists()
    assert len(read_market(market_path(config.storage.root, "coinbase", "ETH/USD", "1h"))) == 150


def test_staging_failure_leaves_no_canonical_or_temp_residue(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    primary = make_synthetic_ohlcv(150, seed=13, exchange="coinbase")
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    providers = _providers(primary)

    real_stage = ingest_module.stage_market
    state = {"count": 0}

    def crashing_stage(path, frame):
        state["count"] += 1
        if state["count"] == 2:
            raise OSError("disk full while staging the second venue")
        return real_stage(path, frame)

    monkeypatch.setattr(ingest_module, "stage_market", crashing_stage)
    with pytest.raises(OSError, match="disk full"):
        ingest_market_data(config, providers=providers)
    raw_root = config.storage.root / "raw"
    assert not market_path(config.storage.root, "coinbase", "ETH/USD", "1h").exists()
    assert not market_path(config.storage.root, "kraken", "ETH/USD", "1h").exists()
    assert not (config.storage.root / ".ares-commit-journal.json").exists()
    assert not list(raw_root.rglob(f"*{TEMP_SUFFIX}")) if raw_root.exists() else True


def test_cross_process_ingestion_lock_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.storage.root.mkdir(parents=True)
    lock_file = config.storage.root / ".ares-ingest.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                from filelock import FileLock
                lock = FileLock({str(lock_file)!r})
                lock.acquire()
                print("locked", flush=True)
                time.sleep(30)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(AresError, match="already running"):
            ingest_market_data(config, providers={})
    finally:
        holder.kill()
        holder.wait()


def test_new_rows_reports_actual_additions_not_fetch_size(tmp_path: Path) -> None:
    config = _config(tmp_path)
    primary = make_synthetic_ohlcv(120, seed=14, exchange="coinbase")
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    providers = _providers(primary)
    first = ingest_market_data(config, providers=providers)
    assert first.exchanges[0].new_rows == 120
    config.data.resume = False  # refetch everything; nothing is actually new
    second = ingest_market_data(config, providers=providers)
    assert second.exchanges[0].new_rows == 0
    hashes = [
        hashlib.sha256(
            market_path(config.storage.root, venue, "ETH/USD", "1h").read_bytes()
        ).hexdigest()
        for venue in ("coinbase", "kraken")
    ]
    third = ingest_market_data(config, providers=providers)
    assert third.committed
    hashes_after = [
        hashlib.sha256(
            market_path(config.storage.root, venue, "ETH/USD", "1h").read_bytes()
        ).hexdigest()
        for venue in ("coinbase", "kraken")
    ]
    assert hashes == hashes_after


def test_non_advancing_cursor_raises_when_duplicates_survive_filtering() -> None:
    """Venues that keep answering with rows at or after the cursor must trip the
    explicit non-advancing-cursor guard instead of looping forever."""
    from datetime import UTC, datetime
    from unittest import mock

    from ares_engine.data.providers import CCXTOHLCVProvider

    start = datetime(2025, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)
    hour_ms = 3_600_000
    same_batch = [[start_ms, 100.0, 101.0, 99.0, 100.5, 5.0]]

    class LoopingExchange:
        has = {"fetchOHLCV": True}
        markets = {"ETH/USD": {}}

        def load_markets(self):
            return None

        def milliseconds(self):
            return start_ms + 100 * hour_ms

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            return same_batch  # never advances past the first candle

    from ares_engine.exceptions import AresError

    with mock.patch.object(CCXTOHLCVProvider, "_exchange", lambda self: LoopingExchange()):
        provider = CCXTOHLCVProvider("coinbase")
        with pytest.raises(AresError, match="non-advancing"):
            provider.fetch_range("ETH/USD", "1h", start, None, limit=10, max_pages=5, retries=0)


def test_non_advancing_cursor_error_message() -> None:
    from datetime import UTC, datetime
    from unittest import mock

    import pytest as _pytest

    from ares_engine.data.providers import CCXTOHLCVProvider
    from ares_engine.exceptions import AresError

    start = datetime(2025, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * 1000)

    class StallingExchange:
        has = {"fetchOHLCV": True}
        markets = {"ETH/USD": {}}
        calls = 0

        def load_markets(self):
            return None

        def milliseconds(self):
            return start_ms + 360_000_000

        def fetch_ohlcv(self, symbol, timeframe, since, limit):
            # Always claims the newest candle is one step BEHIND the cursor.
            return [[since - 3_600_000, 100.0, 101.0, 99.0, 100.5, 5.0]]

    with mock.patch.object(CCXTOHLCVProvider, "_exchange", lambda self: StallingExchange()):
        provider = CCXTOHLCVProvider("coinbase")
        with _pytest.raises(AresError, match="non-advancing"):
            provider.fetch_range("ETH/USD", "1h", start, None, limit=10, max_pages=5, retries=0)
