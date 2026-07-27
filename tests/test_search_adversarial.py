"""Optuna selection integrity: bad candidates must be unable to win."""

import math
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from ares_engine.config import load_config
from ares_engine.search import _trial_config, run_search
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.validation import robust_score


def _summary(score, passed=True):
    return SimpleNamespace(
        passed=passed,
        aggregate={"median_sharpe": score, "note": "stub"},
        gates={"median_sharpe": passed},
        score=score,
    )


def _base_config(tmp_path: Path, name: str, trials: int):
    config = load_config("configs/smoke.yaml")
    config.search.n_trials = trials
    config.search.study_name = name
    config.search.storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    return config


def test_nan_and_infinite_scores_can_never_win(tmp_path, monkeypatch) -> None:
    scores = iter([float("nan"), float("inf"), 0.25, -0.5])

    def fake_walk_forward(frame, config, verbose=0):
        return _summary(next(scores))

    monkeypatch.setattr("ares_engine.search.run_walk_forward", fake_walk_forward)
    config = _base_config(tmp_path, "nan-cannot-win", 4)
    study, best_config = run_search(
        make_synthetic_ohlcv(300), config, output_dir=tmp_path / "artifacts"
    )
    selected = int(study.user_attrs["ares_selected_trial_number"])
    selected_trial = next(t for t in study.trials if t.number == selected)
    assert selected_trial.value == pytest.approx(0.25), (
        "the finite 0.25 trial must beat NaN and +inf candidates"
    )
    assert math.isfinite(float(study.user_attrs["ares_selected_trial_value"]))


def test_pruned_and_crashed_trials_cannot_win(tmp_path, monkeypatch) -> None:
    behaviors = iter(["prune", "crash", 0.1])

    def fake_walk_forward(frame, config, verbose=0):
        behavior = next(behaviors)
        if behavior == "prune":
            raise optuna.TrialPruned()
        if behavior == "crash":
            raise ValueError("degenerate fold")
        return _summary(behavior)

    monkeypatch.setattr("ares_engine.search.run_walk_forward", fake_walk_forward)
    config = _base_config(tmp_path, "pruned-cannot-win", 3)
    study, _ = run_search(make_synthetic_ohlcv(300), config, output_dir=tmp_path / "artifacts")
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.PRUNED in states
    selected = int(study.user_attrs["ares_selected_trial_number"])
    selected_trial = next(t for t in study.trials if t.number == selected)
    assert selected_trial.state == optuna.trial.TrialState.COMPLETE
    assert selected_trial.value == pytest.approx(0.1)


def test_gate_failing_high_scores_lose_to_gate_passing_low_scores(tmp_path, monkeypatch) -> None:
    summaries = iter([_summary(99.0, passed=False), _summary(0.05, passed=True)])
    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *a, **k: next(summaries))
    config = _base_config(tmp_path, "gates-beat-scores", 2)
    study, _ = run_search(make_synthetic_ohlcv(300), config, output_dir=tmp_path / "artifacts")
    selected = int(study.user_attrs["ares_selected_trial_number"])
    selected_trial = next(t for t in study.trials if t.number == selected)
    assert selected_trial.user_attrs["passed"] is True
    assert selected_trial.value == pytest.approx(0.05)


def test_scoring_direction_prefers_higher_scores(tmp_path, monkeypatch) -> None:
    summaries = iter([_summary(0.3), _summary(0.9), _summary(0.6)])
    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *a, **k: next(summaries))
    config = _base_config(tmp_path, "direction", 3)
    study, _ = run_search(make_synthetic_ohlcv(300), config, output_dir=tmp_path / "artifacts")
    assert float(study.user_attrs["ares_selected_trial_value"]) == pytest.approx(0.9)


def test_winner_can_be_rebuilt_exactly_from_frozen_trial(tmp_path, monkeypatch) -> None:
    summaries = iter([_summary(0.2), _summary(0.8)])
    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *a, **k: next(summaries))
    config = _base_config(tmp_path, "rebuild", 2)
    study, best_config = run_search(
        make_synthetic_ohlcv(300), config, output_dir=tmp_path / "artifacts"
    )
    selected = int(study.user_attrs["ares_selected_trial_number"])
    frozen = next(t for t in study.trials if t.number == selected)
    rebuilt = _trial_config(config, frozen)
    assert rebuilt.model_dump(mode="json") == best_config.model_dump(mode="json")
    assert rebuilt.validation.purge_bars == rebuilt.labels.horizon_bars
    meta = (tmp_path / "artifacts" / "best_meta.json").read_text(encoding="utf-8")
    assert '"aggregate"' in meta and '"gates"' in meta


def test_robust_score_penalty_directions() -> None:
    base = {
        "median_sharpe": 1.0,
        "median_return": 0.10,
        "mean_auc": 0.55,
        "worst_drawdown": 0.10,
        "sharpe_std": 0.20,
    }
    score = robust_score(base)
    assert robust_score({**base, "worst_drawdown": 0.30}) < score
    assert robust_score({**base, "sharpe_std": 0.60}) < score
    assert robust_score({**base, "median_sharpe": 1.5}) > score
    assert robust_score({**base, "median_return": 0.20}) > score
    assert robust_score({**base, "mean_auc": 0.65}) > score


def test_changed_dataset_cannot_inherit_stale_trials(tmp_path, monkeypatch) -> None:
    """Regression for the reproduced stale-vintage defect: with a fixed study name,
    a rerun on different data used to select the previous dataset's trial."""
    config = _base_config(tmp_path, "vintage", 1)
    old_data = make_synthetic_ohlcv(300, seed=1)
    new_data = make_synthetic_ohlcv(300, seed=2)

    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *a, **k: _summary(5.0))
    study_old, _ = run_search(old_data, config, output_dir=tmp_path / "a1")
    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *a, **k: _summary(0.1))
    study_new, _ = run_search(new_data, config, output_dir=tmp_path / "a2")

    assert study_old.study_name != study_new.study_name
    assert float(study_new.user_attrs["ares_selected_trial_value"]) == pytest.approx(0.1)
    assert len(study_new.trials) == 1

    # Identical data resumes the same study and may accumulate trials.
    monkeypatch.setattr("ares_engine.search.run_walk_forward", lambda *a, **k: _summary(0.2))
    study_resume, _ = run_search(new_data, config, output_dir=tmp_path / "a3")
    assert study_resume.study_name == study_new.study_name
    assert len(study_resume.trials) == 2
