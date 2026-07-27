from pathlib import Path

import pandas as pd

from ares_engine.cli import _primary_frame
from ares_engine.config import load_config
from ares_engine.synthetic import make_synthetic_ohlcv


def test_primary_frame_revalidates_stored_venues_without_name_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    primary = make_synthetic_ohlcv(300, seed=50, exchange="coinbase")
    secondary = primary.copy()
    secondary["exchange"] = "kraken"
    config.data.since = pd.Timestamp(primary["timestamp"].min()).to_pydatetime()
    config.data.until = (
        pd.Timestamp(primary["timestamp"].max()) + pd.Timedelta(hours=1)
    ).to_pydatetime()

    monkeypatch.setattr("ares_engine.cli.load_config", lambda _: config)
    monkeypatch.setattr("ares_engine.cli.read_market", lambda _: primary)
    monkeypatch.setattr("ares_engine.cli._secondary_frames", lambda _: {"kraken": secondary})

    loaded_config, _, loaded_frame = _primary_frame(Path("config.yaml"))
    assert loaded_config is config
    assert loaded_frame is primary
