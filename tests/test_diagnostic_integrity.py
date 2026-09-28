"""Non-ML counterfactual and independent numeric checks for research diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ares_engine.promotion as promotion
import ares_engine.search as search
import ares_engine.validation as validation
from ares_engine.backtest import run_backtest
from ares_engine.calibration import (
    WALK_FORWARD_PROTOCOL,
    probability_calibration_report,
    threshold_stability_report,
)
from ares_engine.config import AresConfig, BacktestConfig
from ares_engine.dataset import SequenceDataset
from ares_engine.exceptions import BundleIntegrityError
from ares_engine.synthetic import make_synthetic_ohlcv


def _fold(count=6):
    return (
        pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC"),
        np.array([0, 0.01, -0.02, 0.01, 0.005, -0.01])[:count],
        np.array([0.8, 0.2, 0.9, 0.1, 0.6, 0.4])[:count],
    )


@pytest.mark.parametrize("long", [0.52, 0.54, 0.58, 0.70, 0.99])
@pytest.mark.parametrize("short", [0.01, 0.30, 0.42, 0.46, 0.48])
def test_valid_thresholds_retain_exact_base_point(long, short):
    config = BacktestConfig(long_threshold=long, short_threshold=short)
    original = config.model_dump()
    report = threshold_stability_report([_fold()], config, timeframe="1h")
    base = next(point for point in report["points"] if point["delta"] == 0)
    expected = run_backtest(*_fold(), config, timeframe="1h").metrics
    assert base["median_return"] == expected.total_return
    assert base["fold_metrics"] == [expected.to_dict()]
    assert len(report["points"]) + len(report["skipped_points"]) == 5
    assert config.model_dump() == original
    json.dumps(report, allow_nan=False)


def test_floating_point_duplicate_is_not_counted_as_sensitivity():
    report = threshold_stability_report(
        [_fold()], BacktestConfig(), timeframe="1h", deltas=(0.0, 1e-100)
    )
    assert len(report["points"]) == 1
    assert report["points"][0]["delta"] == 0
    assert report["coverage"] == "partial"
    assert not report["sensitivity_available"]
    assert report["skipped_points"][0]["reason"] == "duplicate_effective_thresholds"


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), True, "0.02"])
def test_malformed_deltas_rejected(delta):
    with pytest.raises(ValueError, match="finite numbers"):
        threshold_stability_report([_fold()], BacktestConfig(), timeframe="1h", deltas=(0.0, delta))


@pytest.mark.parametrize("bins", [True, 2.5, 0, 1001])
def test_invalid_bin_count_rejected(bins):
    with pytest.raises(ValueError, match="integer"):
        probability_calibration_report(np.array([0, 1]), np.array([0.2, 0.8]), bin_count=bins)


def test_single_class_and_zero_reference_denominator_are_explicit():
    report = probability_calibration_report(
        np.zeros(3), np.full(3, 0.1), reference_probabilities=np.zeros(3)
    )
    assert "single_class_evaluation" in report["limitations"]
    assert "fewer_samples_than_bins" in report["limitations"]
    assert report["baselines"]["brier_skill_vs_training_prior"] is None
    json.dumps(report, allow_nan=False)


def test_tiny_reference_error_cannot_emit_infinite_skill():
    report = probability_calibration_report(
        np.zeros(2), np.ones(2), reference_probabilities=np.full(2, 1e-160)
    )
    assert report["baselines"]["brier_skill_vs_training_prior"] is None
    assert report["baselines"]["skill_unavailable_reason"]
    json.dumps(report, allow_nan=False)


def test_complex_evidence_is_rejected_without_discarding_imaginary_part():
    with pytest.raises(ValueError, match="real-valued"):
        probability_calibration_report(np.array([0, 1]), np.array([0.2 + 0.4j, 0.8]))


@pytest.mark.parametrize("which", ["labels", "probabilities"])
def test_matrix_is_not_silently_flattened(which):
    inputs = {"labels": np.array([0, 1]), "probabilities": np.array([0.2, 0.8])}
    inputs[which] = inputs[which].reshape(1, 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        probability_calibration_report(**inputs)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "short",
        "matrix",
        "nan",
        "reverse",
        "duplicate",
        "naive",
        "nat",
        "mismatch",
        "bad_return",
    ],
)
def test_invalid_fold_evidence_rejected(case):
    stamps, returns, probabilities = _fold()
    if case == "empty":
        stamps, returns, probabilities = stamps[:0], returns[:0], probabilities[:0]
    if case == "short":
        stamps, returns, probabilities = stamps[:1], returns[:1], probabilities[:1]
    if case == "matrix":
        probabilities = probabilities.reshape(2, 3)
    if case == "nan":
        returns[1] = np.nan
    if case == "reverse":
        stamps = stamps[::-1]
    if case == "duplicate":
        stamps = pd.DatetimeIndex([stamps[0]] * 6)
    if case == "naive":
        stamps = stamps.tz_localize(None)
    if case == "nat":
        stamps = pd.DatetimeIndex([pd.NaT] * 6, tz="UTC")
    if case == "mismatch":
        probabilities = probabilities[:-1]
    if case == "bad_return":
        returns[1] = -1.01
    with pytest.raises(ValueError):
        threshold_stability_report(
            [(stamps, returns, probabilities)], BacktestConfig(), timeframe="1h"
        )


def test_overflowed_metrics_cannot_claim_solvent_stability():
    stamps = _fold(3)[0]
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="non-finite"),
    ):
        threshold_stability_report(
            [(stamps, np.array([0.0, 1e308, 1e308]), np.ones(3))], BacktestConfig(), timeframe="1h"
        )


def test_baselines_share_delay_fees_slippage_and_stress_costs():
    stamps = _fold(3)[0]
    returns = np.array([0.0, 0.01, -0.02])
    config = BacktestConfig(fee_bps=10, slippage_bps=5, execution_delay_bars=1)
    report = threshold_stability_report(
        [(stamps, returns, np.ones(3))], config, timeframe="1h", stress_cost_multiplier=3
    )
    baselines = report["benchmarks"]
    # One delayed entry, then two held bars. No implicit exit charge.
    assert baselines["buy_and_hold"]["base"]["median_return"] == pytest.approx(
        (1 + 0.01 - 0.0015) * (1 - 0.02) - 1
    )
    assert baselines["buy_and_hold"]["stress"]["median_return"] == pytest.approx(
        (1 + 0.01 - 0.0045) * (1 - 0.02) - 1
    )
    assert baselines["always_short"]["base"]["median_return"] == pytest.approx(
        (1 - 0.01 - 0.0015) * (1 + 0.02) - 1
    )
    assert baselines["flat_cash"]["base"]["median_return"] == 0
    assert baselines["flat_cash"]["base"]["total_trades"] == 0
    assert report["points"][0]["stress"] == baselines["buy_and_hold"]["stress"]


def _run_mocked(
    monkeypatch,
    *,
    label_flip=False,
    feature_shift=False,
    overlap=False,
    source_shift=False,
    source_change_during_fit=False,
):
    config = AresConfig()
    config.model.lookback_bars = 4
    config.labels.horizon_bars = 3
    config.validation.purge_bars = 3
    config.validation.min_train_bars = 100
    config.validation.validation_bars = 60
    config.validation.step_bars = 30
    config.validation.max_folds = 2 if overlap else 1
    frame = make_synthetic_ohlcv(240, seed=71)
    if source_shift:
        frame.loc[0, "volume"] += 1
    features = np.repeat(np.arange(210.0)[:, None, None], 4, axis=1)
    labels = np.where(np.arange(210) % 2, 1.0, -1.0)
    if label_flip:
        labels[103:163] *= -1
    if feature_shift:
        features[103:163] += 50
    dataset = SequenceDataset(
        features,
        labels,
        pd.date_range("2026-01-01", periods=210, freq="h", tz="UTC"),
        np.full(210, 100.0),
        np.tile([0.001, -0.001], 105),
        np.arange(210),
        ["row_id"],
    )
    captured = {"fits": [], "predictions": [], "clear": 0}

    def fit(model, X, y, model_config, **kwargs):
        captured["fits"].append((X.copy(), y.copy(), kwargs))
        if source_change_during_fit:
            frame.loc[0, "volume"] += 1

    def predict(model, X):
        values = 0.5 + 0.2 * np.tanh(X[:, 0, 0])
        captured["predictions"].append(values.copy())
        return values

    def clear():
        captured["clear"] += 1

    monkeypatch.setattr(validation, "prepare_dataset", lambda *args: (None, dataset))
    monkeypatch.setattr(validation, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(validation, "fit_model", fit)
    monkeypatch.setattr(validation, "predict_probabilities", predict)
    monkeypatch.setattr(validation, "clear_session", clear)
    return validation.run_walk_forward(frame, config), captured


def test_outer_labels_never_select_weights_or_training_prior(monkeypatch):
    baseline, before = _run_mocked(monkeypatch)
    flipped, after = _run_mocked(monkeypatch, label_flip=True)
    X, y, kwargs = before["fits"][0]
    assert kwargs.get("X_validation") is None and kwargs.get("y_validation") is None
    np.testing.assert_array_equal(X, after["fits"][0][0])
    np.testing.assert_array_equal(y, after["fits"][0][1])
    np.testing.assert_array_equal(before["predictions"][0], after["predictions"][0])
    assert before["clear"] == after["clear"] == 1
    a = baseline.diagnostic_provenance["partitions"][0]
    b = flipped.diagnostic_provenance["partitions"][0]
    assert a["training_positive_rate"] == b["training_positive_rate"] == 0.5
    assert a["training_labels_sha256"] == b["training_labels_sha256"]
    assert a["evaluation_labels_sha256"] != b["evaluation_labels_sha256"]
    assert a["last_training_label_source_row"] < a["first_evaluation_source_row"]


def test_feature_source_and_partition_evidence_are_reproducible_and_sensitive(monkeypatch):
    baseline, captured = _run_mocked(monkeypatch)
    repeated, _ = _run_mocked(monkeypatch)
    assert baseline.to_dict() == repeated.to_dict()
    shifted, changed = _run_mocked(monkeypatch, feature_shift=True)
    np.testing.assert_array_equal(captured["fits"][0][0], changed["fits"][0][0])
    a = baseline.diagnostic_provenance["partitions"][0]
    b = shifted.diagnostic_provenance["partitions"][0]
    assert a["train_features_sha256"] == b["train_features_sha256"]
    assert a["evaluation_features_sha256"] != b["evaluation_features_sha256"]
    source_changed, _ = _run_mocked(monkeypatch, source_shift=True)
    assert (
        baseline.diagnostic_provenance["source_sha256"]
        != source_changed.diagnostic_provenance["source_sha256"]
    )
    assert (
        baseline.diagnostic_provenance["partitions"]
        == source_changed.diagnostic_provenance["partitions"]
    )
    json.dumps(baseline.to_dict(), allow_nan=False)


def test_overlapping_folds_disclose_exact_observation_counts(monkeypatch):
    result, _ = _run_mocked(monkeypatch, overlap=True)
    assert result.probability_calibration["sample_count"] == 120
    assert result.probability_calibration["unique_sample_count"] == 90
    assert result.probability_calibration["repeated_observation_count"] == 30
    assert result.threshold_stability["observation_count"] == 120
    assert result.threshold_stability["unique_timestamp_count"] == 90


def test_fitting_uses_owned_source_snapshot(monkeypatch):
    baseline, _ = _run_mocked(monkeypatch)
    mutated, _ = _run_mocked(monkeypatch, source_change_during_fit=True)
    assert baseline.to_dict() == mutated.to_dict()


def test_fit_protocol_change_isolates_resumed_study(monkeypatch):
    frame, config = make_synthetic_ohlcv(240), AresConfig()
    current = search._data_fingerprint(frame, config)
    monkeypatch.setattr(search, "WALK_FORWARD_PROTOCOL", "legacy_outer_validation_auc")
    assert search._data_fingerprint(frame, config) != current


@pytest.mark.parametrize("new_challenger", [True, False])
def test_promotion_never_compares_legacy_and_current_fit_scores(monkeypatch, new_challenger):
    old = {"passed": True, "score": 1.0}
    new = {
        "passed": True,
        "score": 999.0,
        "diagnostic_provenance": {"fit_protocol": WALK_FORWARD_PROTOCOL},
    }
    metrics = [new, old] if new_challenger else [old, new]
    monkeypatch.setattr(promotion, "verify_bundle", lambda _: None)
    monkeypatch.setattr(promotion, "_metrics", lambda path: metrics[0 if path.name == "new" else 1])
    decision = promotion.decide_promotion(Path("new"), Path("old"), AresConfig().gates)
    assert not decision.approved
    assert decision.improvement is None
    assert "protocol mismatch" in decision.reason


@pytest.mark.parametrize("provenance", [None, {}, {"fit_protocol": "unknown"}])
def test_malformed_explicit_protocol_cannot_be_first_champion(monkeypatch, provenance):
    monkeypatch.setattr(promotion, "verify_bundle", lambda _: None)
    monkeypatch.setattr(
        promotion,
        "_metrics",
        lambda _: {"passed": True, "score": 1.0, "diagnostic_provenance": provenance},
    )
    with pytest.raises(BundleIntegrityError, match="unsupported fitting protocol"):
        promotion.decide_promotion(Path("candidate"), None, AresConfig().gates)


def test_stress_bankruptcy_is_not_a_solvent_summary():
    report = threshold_stability_report(
        [(_fold(3)[0], np.array([0.0, -0.999, 0.0]), np.ones(3))], BacktestConfig(), timeframe="1h"
    )
    assert report["summary"]["all_evaluated_points_solvent_at_base_cost"]
    assert not report["summary"]["all_evaluated_points_solvent_under_stress"]
    assert not report["summary"]["all_points_solvable"]


def test_real_preparation_preserves_directional_alignment_and_training_prior(monkeypatch):
    config = AresConfig()
    config.model.lookback_bars = 4
    config.validation.min_train_bars = 150
    config.validation.validation_bars = 100
    config.validation.max_folds = 1
    frame = make_synthetic_ohlcv(500, seed=72)
    _, dataset = validation.prepare_dataset(frame, config)
    fold = next(
        validation.PurgedWalkForwardSplitter(
            min_train_size=150,
            validation_size=100,
            step_size=100,
            purge_size=config.labels.horizon_bars,
            max_splits=1,
        ).split(len(dataset.X))
    )
    train = fold.train_indices[dataset.directional_mask[fold.train_indices]]
    evaluation = fold.validation_indices[dataset.directional_mask[fold.validation_indices]]
    assert 0 < len(evaluation) < 100
    probabilities = np.linspace(0.1, 0.9, 100)
    monkeypatch.setattr(validation, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(validation, "fit_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, "clear_session", lambda: None)
    monkeypatch.setattr(validation, "predict_probabilities", lambda *args: probabilities)
    result = validation.run_walk_forward(frame, config)
    positions = np.searchsorted(fold.validation_indices, evaluation)
    y = dataset.y_binary[evaluation]
    assert result.probability_calibration["sample_count"] == len(evaluation)
    assert result.probability_calibration["brier_score"] == pytest.approx(
        np.mean((probabilities[positions] - y) ** 2)
    )
    expected_prior = float(dataset.y_binary[train].astype("float32").mean())
    assert result.probability_calibration["baselines"]["training_prior_brier"] == pytest.approx(
        np.mean((expected_prior - y) ** 2)
    )


@pytest.mark.parametrize("failed_operation", ["fit_model", "predict_probabilities"])
def test_model_cleanup_after_failure(monkeypatch, failed_operation):
    config = AresConfig()
    config.model.lookback_bars = 4
    config.validation.min_train_bars = 150
    config.validation.validation_bars = 100
    config.validation.max_folds = 1
    cleared = []
    monkeypatch.setattr(validation, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(validation, "fit_model", lambda *args, **kwargs: None)

    def fail(*args, **kwargs):
        raise RuntimeError("injected model failure")

    monkeypatch.setattr(validation, failed_operation, fail)
    monkeypatch.setattr(validation, "clear_session", lambda: cleared.append(True))
    with pytest.raises(RuntimeError, match="injected"):
        validation.run_walk_forward(make_synthetic_ohlcv(500, seed=72), config)
    assert cleared == [True]
