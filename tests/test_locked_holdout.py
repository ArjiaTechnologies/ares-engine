"""Adversarial guarantees for the one-time post-search final holdout."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock

from ares_engine import holdout
from ares_engine.config import AresConfig, load_config
from ares_engine.search import SEARCH_SPACE, _data_fingerprint
from ares_engine.synthetic import make_synthetic_ohlcv


def _config() -> AresConfig:
    config = load_config("configs/smoke.yaml")
    config.validation.min_train_bars = 100
    config.validation.validation_bars = 50
    config.validation.max_folds = 1
    config.model.lookback_bars = 12
    config.model.epochs = 1
    return config


def _frame(seed: int = 27) -> pd.DataFrame:
    return make_synthetic_ohlcv(360, seed=seed, exchange="coinbase")


def _result() -> dict[str, Any]:
    return {
        "sample_count": 72,
        "candidate": {"solvent": True, "backtest": {"total_return": 0.01}},
        "benchmarks": {
            "flat_cash": {"backtest": {"total_return": 0.0}},
            "buy_and_hold": {"backtest": {"total_return": 0.02}},
            "lagged_momentum": {"backtest": {"total_return": -0.01}},
        },
    }


def _stub_search(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    seen: list[pd.DataFrame] = []

    def search(
        frame: pd.DataFrame,
        config: AresConfig,
        *,
        output_dir: Path,
        verbose: int,
    ) -> tuple[SimpleNamespace, AresConfig]:
        seen.append(frame.copy(deep=True))
        study = SimpleNamespace(
            user_attrs={
                "ares_data_fingerprint": _data_fingerprint(frame, config),
                "ares_selected_trial_number": 3,
            }
        )
        return study, config.model_copy(deep=True)

    monkeypatch.setattr(holdout, "run_search", search)
    monkeypatch.setattr(holdout, "_evaluate_holdout", lambda *args, **kwargs: _result())
    return seen


def test_holdout_is_quarantined_and_commitment_is_consumed_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen = _stub_search(monkeypatch)
    frame = _frame()
    config = _config()
    result = holdout.run_locked_holdout(
        frame,
        config,
        holdout_bars=80,
        output_dir=tmp_path,
    )

    embargo = max(int(value) for value in SEARCH_SPACE["horizon_bars"])
    assert len(seen) == 1
    assert len(seen[0]) == len(frame) - 80 - embargo
    assert seen[0]["timestamp"].max() < frame.iloc[-80]["timestamp"]
    assert result["research_dataset_sha256"] == _data_fingerprint(seen[0], config)
    assert result["holdout_dataset_sha256"] != result["research_dataset_sha256"]
    assert result["one_time_evaluation"] is True
    assert result["profitability_established"] is False
    assert result["order_execution_enabled"] is False
    commitment = json.loads((tmp_path / "commitment.json").read_text(encoding="utf-8"))
    assert commitment["state"] == "EVALUATION_COMPLETE"
    assert commitment["report_sha256"] == result["report_sha256"]

    with pytest.raises(holdout.HoldoutIntegrityError, match="holdout_already_consumed"):
        holdout.run_locked_holdout(frame, config, holdout_bars=80, output_dir=tmp_path)
    assert len(seen) == 1


def test_future_holdout_mutation_cannot_change_any_search_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen = _stub_search(monkeypatch)
    original = _frame()
    mutated = original.copy(deep=True)
    mutated.loc[mutated.index[-80:], ["open", "high", "low", "close"]] *= 3.0
    config = _config()

    first = holdout.run_locked_holdout(original, config, holdout_bars=80, output_dir=tmp_path / "a")
    second = holdout.run_locked_holdout(mutated, config, holdout_bars=80, output_dir=tmp_path / "b")

    pd.testing.assert_frame_equal(seen[0], seen[1])
    assert first["research_dataset_sha256"] == second["research_dataset_sha256"]
    assert first["holdout_dataset_sha256"] != second["holdout_dataset_sha256"]


def test_failed_final_inspection_is_consumed_and_cannot_be_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_search(monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("model evaluation crashed")

    monkeypatch.setattr(holdout, "_evaluate_holdout", fail)
    with pytest.raises(RuntimeError, match="model evaluation crashed"):
        holdout.run_locked_holdout(_frame(), _config(), holdout_bars=80, output_dir=tmp_path)
    commitment = json.loads((tmp_path / "commitment.json").read_text(encoding="utf-8"))
    assert commitment["state"] == "EVALUATION_FAILED"
    assert not (tmp_path / "final_report.json").exists()

    monkeypatch.setattr(holdout, "_evaluate_holdout", lambda *args, **kwargs: _result())
    with pytest.raises(holdout.HoldoutIntegrityError, match="holdout_already_consumed"):
        holdout.run_locked_holdout(_frame(), _config(), holdout_bars=80, output_dir=tmp_path)


def test_short_embargo_cannot_follow_only_the_current_trial_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_search(monkeypatch)
    with pytest.raises(holdout.HoldoutIntegrityError, match="maximum_search_horizon"):
        holdout.run_locked_holdout(
            _frame(),
            _config(),
            holdout_bars=80,
            embargo_bars=6,
            output_dir=tmp_path,
        )


def test_research_configuration_mutation_fails_before_holdout_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evaluated: list[bool] = []

    def mutate(
        frame: pd.DataFrame,
        config: AresConfig,
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, AresConfig]:
        config.project.seed += 1
        return SimpleNamespace(user_attrs={"ares_selected_trial_number": 1}), config

    monkeypatch.setattr(holdout, "run_search", mutate)
    monkeypatch.setattr(holdout, "_evaluate_holdout", lambda *args, **kwargs: evaluated.append(True))
    with pytest.raises(holdout.HoldoutIntegrityError, match="configuration_mutated_during_search"):
        holdout.run_locked_holdout(_frame(), _config(), holdout_bars=80, output_dir=tmp_path)
    assert evaluated == []


def test_wrong_study_partition_and_unfrozen_winner_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = (
        ({"ares_data_fingerprint": "wrong", "ares_selected_trial_number": 1}, "fingerprint"),
        ({"ares_data_fingerprint": "correct"}, "winner_not_frozen"),
    )
    for index, (attributes, blocker) in enumerate(cases):

        def invalid_search(
            frame: pd.DataFrame,
            config: AresConfig,
            **kwargs: Any,
        ) -> tuple[SimpleNamespace, AresConfig]:
            values = dict(attributes)
            if values.get("ares_data_fingerprint") == "correct":
                values["ares_data_fingerprint"] = _data_fingerprint(frame, config)
            return SimpleNamespace(user_attrs=values), config.model_copy(deep=True)

        monkeypatch.setattr(holdout, "run_search", invalid_search)
        with pytest.raises(holdout.HoldoutIntegrityError, match=blocker):
            holdout.run_locked_holdout(
                _frame(),
                _config(),
                holdout_bars=80,
                output_dir=tmp_path / str(index),
            )


def test_insolvent_and_nonfinite_final_results_consume_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for index, (mutate, blocker) in enumerate(
        (
            (lambda result: result["candidate"].update(solvent=False), "insolvent"),
            (lambda result: result["candidate"]["backtest"].update(total_return=float("nan")), "non_finite"),
        )
    ):
        _stub_search(monkeypatch)

        def invalid_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = _result()
            mutate(result)
            return result

        monkeypatch.setattr(holdout, "_evaluate_holdout", invalid_result)
        directory = tmp_path / str(index)
        with pytest.raises(holdout.HoldoutIntegrityError, match=blocker):
            holdout.run_locked_holdout(_frame(), _config(), holdout_bars=80, output_dir=directory)
        commitment = json.loads((directory / "commitment.json").read_text(encoding="utf-8"))
        assert commitment["state"] == "EVALUATION_FAILED"


def test_competing_evaluation_fails_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_search(monkeypatch)
    with FileLock(str(tmp_path / ".holdout.lock")):
        with pytest.raises(holdout.HoldoutIntegrityError, match="already_running"):
            holdout.run_locked_holdout(_frame(), _config(), holdout_bars=80, output_dir=tmp_path)


def test_real_benchmarks_and_early_stopping_use_only_research_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    config = _config()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(holdout, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(holdout, "clear_session", lambda: None)

    def record_fit(
        model: object,
        features: np.ndarray,
        labels: np.ndarray,
        model_config: object,
        **kwargs: Any,
    ) -> None:
        captured["training"] = features.copy()
        captured["early_stopping"] = kwargs["X_validation"].copy()

    monkeypatch.setattr(holdout, "fit_model", record_fit)
    monkeypatch.setattr(
        holdout,
        "predict_probabilities",
        lambda model, features: np.full(len(features), 0.65, dtype="float64"),
    )

    result = holdout._evaluate_holdout(
        frame,
        config,
        research_end=len(frame) - 80 - 24,
        holdout_start=len(frame) - 80,
        verbose=0,
    )
    first_training = captured["training"].copy()
    first_early_stopping = captured["early_stopping"].copy()
    mutated = frame.copy(deep=True)
    mutated.loc[mutated.index[-80:], ["open", "high", "low", "close"]] *= 2.0
    holdout._evaluate_holdout(
        mutated,
        config,
        research_end=len(frame) - 80 - 24,
        holdout_start=len(frame) - 80,
        verbose=0,
    )

    np.testing.assert_array_equal(first_training, captured["training"])
    np.testing.assert_array_equal(first_early_stopping, captured["early_stopping"])
    assert set(result["benchmarks"]) == {"flat_cash", "buy_and_hold", "lagged_momentum"}
    assert result["benchmarks"]["flat_cash"]["backtest"]["total_return"] == 0.0
    assert result["candidate"]["solvent"] is True
    assert result["sample_count"] >= 30
