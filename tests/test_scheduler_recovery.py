"""Scheduler cycle safety: visibility of failures, idempotency, lock hygiene, log safety."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import ares_engine.scheduler as scheduler_module
from ares_engine.config import load_config
from ares_engine.exceptions import DataQualityError, PromotionRejected
from ares_engine.scheduler import _cycle_lock, deep_cycle, quick_cycle


def _cycle_config(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config.storage.root = tmp_path / "data"
    config.storage.artifacts = tmp_path / "artifacts"
    return config


def test_quick_cycle_without_champion_ends_visibly_after_ingestion(
    tmp_path, monkeypatch, caplog
) -> None:
    config = _cycle_config(tmp_path)
    monkeypatch.setattr(scheduler_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        scheduler_module,
        "ingest_market_data",
        lambda _: SimpleNamespace(passed=True, committed=True),
    )
    monkeypatch.setattr(scheduler_module, "resolve_champion", lambda _: None)
    called = {"paper": False}
    monkeypatch.setattr(
        scheduler_module, "generate_paper_signal", lambda *a, **k: called.__setitem__("paper", True)
    )
    with caplog.at_level(logging.WARNING):
        quick_cycle(Path("config.yaml"))
    assert called["paper"] is False
    assert any("No champion exists" in record.message for record in caplog.records)


def test_failed_promotion_propagates_and_is_not_swallowed(tmp_path, monkeypatch) -> None:
    config = _cycle_config(tmp_path)
    config.scheduler.run_search_on_deep = False
    monkeypatch.setattr(scheduler_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        scheduler_module,
        "ingest_market_data",
        lambda _: SimpleNamespace(passed=True, committed=True),
    )
    monkeypatch.setattr(scheduler_module, "read_market", lambda _: object())
    monkeypatch.setattr(
        scheduler_module, "train_candidate", lambda *a, **k: (tmp_path / "bundle", {})
    )

    def rejecting_promote(*args, **kwargs):
        raise PromotionRejected("weaker than incumbent")

    monkeypatch.setattr(scheduler_module, "promote", rejecting_promote)
    with pytest.raises(PromotionRejected, match="weaker"):
        deep_cycle(Path("config.yaml"))


def test_failed_validation_prevents_promotion_call(tmp_path, monkeypatch) -> None:
    config = _cycle_config(tmp_path)
    config.scheduler.run_search_on_deep = False
    monkeypatch.setattr(scheduler_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        scheduler_module,
        "ingest_market_data",
        lambda _: SimpleNamespace(passed=True, committed=True),
    )
    monkeypatch.setattr(scheduler_module, "read_market", lambda _: object())

    def failing_train(*args, **kwargs):
        raise ValueError("Candidate failed hard validation gates: ['drawdown']")

    promoted = {"called": False}
    monkeypatch.setattr(scheduler_module, "train_candidate", failing_train)
    monkeypatch.setattr(
        scheduler_module, "promote", lambda *a, **k: promoted.__setitem__("called", True)
    )
    with pytest.raises(ValueError, match="hard validation gates"):
        deep_cycle(Path("config.yaml"))
    assert promoted["called"] is False


def test_cycle_lock_is_released_after_success_and_after_exceptions(tmp_path) -> None:
    root = tmp_path / "data"
    with _cycle_lock(root):
        pass
    with pytest.raises(RuntimeError, match="boom"), _cycle_lock(root):
        raise RuntimeError("boom")
    # Lock must be reacquirable immediately after both exits.
    with _cycle_lock(root):
        pass


def test_failed_ingestion_stops_quick_cycle_before_any_paper_signal(tmp_path, monkeypatch) -> None:
    config = _cycle_config(tmp_path)
    monkeypatch.setattr(scheduler_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        scheduler_module,
        "ingest_market_data",
        lambda _: SimpleNamespace(
            passed=False, committed=False, exchanges=[], cross_venue_reports=[]
        ),
    )
    touched = {"paper": False}
    monkeypatch.setattr(
        scheduler_module,
        "generate_paper_signal",
        lambda *a, **k: touched.__setitem__("paper", True),
    )
    with pytest.raises(DataQualityError, match="failed ingestion"):
        quick_cycle(Path("config.yaml"))
    assert touched["paper"] is False


def test_scheduler_logs_never_contain_environment_secrets(tmp_path, monkeypatch, caplog) -> None:
    secret = "SUPER-SECRET-EXCHANGE-KEY-12345"
    monkeypatch.setenv("COINBASE_API_KEY", secret)
    config = _cycle_config(tmp_path)
    monkeypatch.setattr(scheduler_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        scheduler_module,
        "ingest_market_data",
        lambda _: SimpleNamespace(passed=True, committed=True),
    )
    monkeypatch.setattr(scheduler_module, "resolve_champion", lambda _: None)
    with caplog.at_level(logging.DEBUG):
        quick_cycle(Path("config.yaml"))
    assert secret not in caplog.text


def test_repeated_quick_cycles_are_idempotent_with_static_data(tmp_path, monkeypatch) -> None:
    config = _cycle_config(tmp_path)
    calls = {"count": 0}

    def counting_ingest(_):
        calls["count"] += 1
        return SimpleNamespace(passed=True, committed=True)

    monkeypatch.setattr(scheduler_module, "load_config", lambda _: config)
    monkeypatch.setattr(scheduler_module, "ingest_market_data", counting_ingest)
    monkeypatch.setattr(scheduler_module, "resolve_champion", lambda _: None)
    quick_cycle(Path("config.yaml"))
    quick_cycle(Path("config.yaml"))
    assert calls["count"] == 2  # each run performs its own locked, gated pass
