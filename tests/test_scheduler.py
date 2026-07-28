from pathlib import Path
from types import SimpleNamespace

import pytest

from ares_engine.config import load_config
from ares_engine.exceptions import DataQualityError
from ares_engine.scheduler import deep_cycle


def test_deep_cycle_runs_search_before_training(monkeypatch, tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.artifacts = tmp_path / "artifacts"
    config.scheduler.run_search_on_deep = True
    best_config = config.model_copy(deep=True)
    best_config.model.family = "tcn"
    calls: list[str] = []
    frame = object()
    bundle = tmp_path / "artifacts" / "candidate"

    monkeypatch.setattr("ares_engine.scheduler.load_config", lambda _: config)
    monkeypatch.setattr(
        "ares_engine.scheduler.ingest_market_data",
        lambda _: calls.append("ingest") or SimpleNamespace(passed=True, committed=True),
    )
    monkeypatch.setattr("ares_engine.scheduler.read_market", lambda _: frame)

    def fake_search(received_frame, received_config):
        assert received_frame is frame
        assert received_config is config
        calls.append("search")
        return object(), best_config

    def fake_train(received_frame, received_config, **kwargs):
        assert received_frame is frame
        assert received_config is best_config
        assert kwargs["source_path"].name == "1h.parquet"
        calls.append("train")
        return bundle, {}

    def fake_promote(received_bundle, artifacts, gates):
        assert received_bundle == bundle
        assert artifacts == best_config.storage.artifacts
        assert gates is best_config.gates
        calls.append("promote")

    monkeypatch.setattr("ares_engine.scheduler.run_search", fake_search)
    monkeypatch.setattr("ares_engine.scheduler.train_candidate", fake_train)
    monkeypatch.setattr("ares_engine.scheduler.promote", fake_promote)

    deep_cycle(Path("config.yaml"))
    assert calls == ["ingest", "search", "train", "promote"]


def test_cycle_lock_fails_closed_when_another_cycle_is_running(tmp_path: Path) -> None:
    from filelock import FileLock

    from ares_engine.scheduler import _cycle_lock

    root = tmp_path / "data"
    root.mkdir(parents=True)
    with FileLock(root / ".ares-cycle.lock"):
        with pytest.raises(RuntimeError, match="already running"):
            with _cycle_lock(root):
                raise AssertionError("lock body must not run")


def test_deep_cycle_stops_when_ingestion_does_not_commit(monkeypatch, tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    calls: list[str] = []
    failed = SimpleNamespace(
        passed=False,
        committed=False,
        exchanges=[],
        cross_venue_reports=[],
    )

    monkeypatch.setattr("ares_engine.scheduler.load_config", lambda _: config)
    monkeypatch.setattr("ares_engine.scheduler.ingest_market_data", lambda _: failed)
    monkeypatch.setattr("ares_engine.scheduler.read_market", lambda _: calls.append("read"))

    with pytest.raises(DataQualityError, match="failed ingestion"):
        deep_cycle(Path("config.yaml"))
    assert calls == []
