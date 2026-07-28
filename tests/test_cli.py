from pathlib import Path

import pandas as pd
import pytest

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
    monkeypatch.setattr(
        "ares_engine.cli._secondary_frames", lambda _, **kwargs: {"kraken": secondary}
    )

    loaded_config, _, loaded_frame = _primary_frame(Path("configs/smoke.yaml"))
    assert loaded_config is config
    assert loaded_frame is primary


def test_packaged_configs_stay_in_sync_with_repository_configs() -> None:
    import importlib.resources
    from pathlib import Path as P

    for name in ["default.yaml", "smoke.yaml"]:
        packaged = (importlib.resources.files("ares_engine") / "configs" / name).read_text(
            encoding="utf-8"
        )
        repo = P("configs", name).read_text(encoding="utf-8")
        assert packaged == repo, f"src/ares_engine/configs/{name} drifted from configs/{name}"


def test_config_resolution_falls_back_to_packaged_copies(tmp_path, monkeypatch) -> None:
    from ares_engine.cli import _resolve_config

    monkeypatch.chdir(tmp_path)  # no ./configs checkout here
    resolved = _resolve_config(Path("configs/smoke.yaml"))
    assert resolved.exists()
    import typer

    with pytest.raises(typer.BadParameter, match="not found"):
        _resolve_config(Path("configs/nonexistent.yaml"))
