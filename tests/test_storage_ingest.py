from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ares_engine.config import load_config
from ares_engine.data.ingest import ingest_market_data
from ares_engine.data.storage import market_path, read_market, sync_duckdb
from ares_engine.synthetic import make_synthetic_ohlcv

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")


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
        del symbol, timeframe, limit, max_pages, retries
        output = self.frame[self.frame["timestamp"] >= pd.Timestamp(since)].copy()
        if until is not None:
            output = output[output["timestamp"] < pd.Timestamp(until)]
        return output


def test_ingest_persists_and_resumes(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.artifacts = tmp_path / "artifacts"
    config.storage.duckdb_path = tmp_path / "data" / "ares.duckdb"
    config.data.since = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    config.data.drop_open_candle = False
    config.data.max_gap_count = 0
    config.data.max_cross_venue_p95_bps = 20

    primary = make_synthetic_ohlcv(500, seed=10, exchange="coinbase")
    config.data.until = (primary["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    secondary["close"] *= 1.0001
    providers = {
        "coinbase": FakeProvider("coinbase", primary),
        "kraken": FakeProvider("kraken", secondary),
    }
    first = ingest_market_data(config, providers=providers)
    second = ingest_market_data(config, providers=providers)
    assert first.passed
    assert second.passed
    assert first.committed
    assert second.committed
    path = market_path(config.storage.root, "coinbase", "ETH/USD", "1h")
    stored = read_market(path)
    assert len(stored) == 500
    assert path.exists()
    assert config.storage.duckdb_path.exists()


def test_duckdb_view_exists_before_first_parquet_file(tmp_path: Path) -> None:
    import duckdb

    database = tmp_path / "data" / "ares.duckdb"
    sync_duckdb(database, tmp_path / "data")
    with duckdb.connect(str(database)) as connection:
        assert connection.execute("SELECT count(*) FROM ohlcv").fetchone()[0] == 0
