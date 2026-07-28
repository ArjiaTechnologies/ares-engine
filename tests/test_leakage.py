"""Adversarial leakage audit: future data must be unable to affect past artifacts."""

import numpy as np
import pandas as pd
import pytest

import ares_engine.validation as validation_module
from ares_engine.config import load_config
from ares_engine.dataset import PurgedWalkForwardSplitter, fit_scaler
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.validation import prepare_dataset, run_walk_forward

pytestmark = pytest.mark.ml


def _small_config():
    config = load_config("configs/smoke.yaml")
    config.project.seed = 123
    config.model.lookback_bars = 12
    config.model.hidden_units = 8
    config.model.epochs = 1
    config.model.patience = 1
    config.model.batch_size = 64
    config.features.ema_periods = [5, 9]
    config.features.rsi_period = 7
    config.features.bollinger_period = 10
    config.features.volatility_windows = [6]
    config.features.volume_z_window = 8
    config.labels.horizon_bars = 4
    config.validation.min_train_bars = 300
    config.validation.validation_bars = 60
    config.validation.step_bars = 60
    config.validation.max_folds = 1
    config.validation.purge_bars = 4
    config.gates.min_median_sharpe = -100.0
    config.gates.min_median_return = -10.0
    config.gates.max_worst_drawdown = 0.999999
    config.gates.max_median_turnover = 100000.0
    config.gates.min_total_trades = 0
    config.gates.require_positive_cost_stress = False
    return config


def _mutate_after(frame: pd.DataFrame, cutoff_row: int) -> pd.DataFrame:
    """Radically rewrite everything at and after cutoff_row."""
    mutated = frame.copy()
    tail = mutated.index[cutoff_row:]
    rng = np.random.default_rng(999)
    mutated.loc[tail, "close"] = rng.uniform(50, 90_000, size=len(tail))
    mutated.loc[tail, "open"] = mutated.loc[tail, "close"] * 0.997
    mutated.loc[tail, "high"] = mutated.loc[tail, ["open", "close"]].max(axis=1) * 1.01
    mutated.loc[tail, "low"] = mutated.loc[tail, ["open", "close"]].min(axis=1) * 0.99
    mutated.loc[tail, "volume"] = rng.uniform(1, 10_000, size=len(tail))
    return mutated


def test_features_labels_tensors_and_scalers_ignore_future_mutation() -> None:
    config = _small_config()
    frame = make_synthetic_ohlcv(500, seed=21)
    cutoff = 400
    base_features, base_dataset = prepare_dataset(frame, config)
    _, mutated_dataset = prepare_dataset(_mutate_after(frame, cutoff), config)

    # Sample rows whose full window, label horizon, and bar return live before the cutoff.
    horizon = config.labels.horizon_bars
    safe = base_dataset.source_rows + horizon < cutoff
    safe_count = int(safe.sum())
    assert safe_count > 200, "fixture must contain a meaningful pre-cutoff region"

    np.testing.assert_array_equal(
        base_dataset.X[safe],
        mutated_dataset.X[:safe_count],
        err_msg="feature tensors before the cutoff must be bitwise identical",
    )
    np.testing.assert_array_equal(
        base_dataset.labels[safe],
        mutated_dataset.labels[:safe_count],
        err_msg="labels whose horizon ends before the cutoff must be identical",
    )
    np.testing.assert_array_equal(
        base_dataset.bar_returns[safe], mutated_dataset.bar_returns[:safe_count]
    )

    train_slice = base_dataset.X[safe][:150]
    scaler_a = fit_scaler(train_slice)
    scaler_b = fit_scaler(mutated_dataset.X[:safe_count][:150])
    np.testing.assert_array_equal(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_array_equal(scaler_a.scale_, scaler_b.scale_)


def test_fold_scalers_are_fit_on_train_indices_only(monkeypatch) -> None:
    config = _small_config()
    frame = make_synthetic_ohlcv(500, seed=22)
    _, dataset = prepare_dataset(frame, config)
    captured: list[np.ndarray] = []
    real_fit_scaler = validation_module.fit_scaler

    def spying_fit_scaler(X_train):
        captured.append(np.asarray(X_train))
        return real_fit_scaler(X_train)

    monkeypatch.setattr(validation_module, "fit_scaler", spying_fit_scaler)
    run_walk_forward(frame, config)

    splitter = PurgedWalkForwardSplitter(
        min_train_size=config.validation.min_train_bars,
        validation_size=config.validation.validation_bars,
        step_size=config.validation.step_bars,
        purge_size=int(config.validation.purge_bars or 0),
        max_splits=config.validation.max_folds,
    )
    folds = list(splitter.split(len(dataset.X)))
    assert len(captured) == len(folds) >= 1
    for fold, seen in zip(folds, captured, strict=True):
        np.testing.assert_array_equal(
            seen,
            dataset.X[fold.train_indices],
            err_msg="scaler input must be exactly the training tensor of the fold",
        )


def test_purge_arithmetic_prevents_label_overlap_with_validation() -> None:
    config = _small_config()
    frame = make_synthetic_ohlcv(500, seed=23)
    _, dataset = prepare_dataset(frame, config)
    horizon = config.labels.horizon_bars
    splitter = PurgedWalkForwardSplitter(
        min_train_size=config.validation.min_train_bars,
        validation_size=config.validation.validation_bars,
        step_size=config.validation.step_bars,
        purge_size=int(config.validation.purge_bars or 0),
        max_splits=5,
    )
    for fold in splitter.split(len(dataset.X)):
        last_train_decision_row = dataset.source_rows[fold.train_indices[-1]]
        first_validation_decision_row = dataset.source_rows[fold.validation_indices[0]]
        # Train labels may use rows up to decision+horizon; that must stay strictly
        # before the first validation decision row.
        assert last_train_decision_row + horizon < first_validation_decision_row
        # And chronological ordering with a purge gap in sample space:
        assert (
            fold.train_indices[-1] + int(config.validation.purge_bars or 0)
            < fold.validation_indices[0]
        )


def test_walk_forward_metrics_are_invariant_to_post_fold_future() -> None:
    """Radically rewriting data after the only fold must not change that fold's results."""
    config = _small_config()
    frame = make_synthetic_ohlcv(420, seed=24)
    # Single fold: train samples end + purge + validation of 60 samples.
    # Sequence samples start at raw row lookback-1=11. Compute the raw row where
    # everything the fold can legally observe ends: validation decision rows end at
    # sample index min_train+purge+60-1 -> raw row +11; labels look horizon ahead;
    # bar returns need one more row. Mutate strictly beyond that boundary.
    _, dataset = prepare_dataset(frame, config)
    last_needed_sample = config.validation.min_train_bars + 4 + 60 - 1
    last_needed_row = int(dataset.source_rows[last_needed_sample]) + config.labels.horizon_bars + 1
    cutoff = last_needed_row + 1
    assert cutoff < len(frame), "fixture must leave future rows to mutate"

    baseline_first = run_walk_forward(frame, config)
    baseline_second = run_walk_forward(frame, config)
    assert baseline_first.to_dict() == baseline_second.to_dict(), (
        "training is not deterministic under the configured seed; leakage checks "
        "cannot be trusted and the reproducibility promise is broken"
    )

    mutated = run_walk_forward(_mutate_after(frame, cutoff), config)
    base_fold = baseline_first.folds[0]
    mutated_fold = mutated.folds[0]
    assert base_fold.to_dict() == mutated_fold.to_dict(), (
        "fold metrics changed when only post-fold future data was rewritten: "
        "information from after the fold is leaking into training or evaluation"
    )


def test_validation_probabilities_are_invariant_to_future_mutation(monkeypatch) -> None:
    config = _small_config()
    frame = make_synthetic_ohlcv(420, seed=25)
    _, dataset = prepare_dataset(frame, config)
    last_needed_sample = config.validation.min_train_bars + 4 + 60 - 1
    cutoff = int(dataset.source_rows[last_needed_sample]) + config.labels.horizon_bars + 2

    captured: list[np.ndarray] = []
    real_predict = validation_module.predict_probabilities

    def spying_predict(model, X):
        result = real_predict(model, X)
        captured.append(np.asarray(result, dtype="float64"))
        return result

    monkeypatch.setattr(validation_module, "predict_probabilities", spying_predict)
    run_walk_forward(frame, config)
    baseline = [array.copy() for array in captured]
    captured.clear()
    run_walk_forward(_mutate_after(frame, cutoff), config)
    assert len(baseline) == len(captured) >= 1
    for before, after in zip(baseline, captured, strict=True):
        np.testing.assert_array_equal(
            before,
            after,
            err_msg="per-fold validation probabilities must not react to future rewrites",
        )
