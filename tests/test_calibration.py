from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ares_engine.calibration import probability_calibration_report, threshold_stability_report
from ares_engine.config import BacktestConfig


def test_probability_calibration_reports_exact_reliability_metrics() -> None:
    report = probability_calibration_report(
        np.asarray([0, 0, 1, 1]), np.asarray([0.0, 0.2, 0.8, 1.0]), bin_count=5
    )
    assert report["sample_count"] == 4
    assert report["positive_count"] == 2
    assert report["brier_score"] == pytest.approx(0.02)
    assert report["expected_calibration_error"] == pytest.approx(0.1)
    assert sum(item["count"] for item in report["bins"]) == 4
    assert report["bins"][-1]["upper_bound"] == 1.0


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [
        ([0], [float("nan")]),
        ([0], [-0.01]),
        ([1], [1.01]),
        ([2], [0.5]),
        ([], []),
        ([0, 1], [0.5]),
    ],
)
def test_probability_calibration_fails_closed_on_invalid_evidence(labels, probabilities) -> None:
    with pytest.raises(ValueError):
        probability_calibration_report(np.asarray(labels), np.asarray(probabilities))


def test_threshold_stability_reuses_predictions_and_exposes_nearby_behavior() -> None:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    probabilities = np.asarray([0.1, 0.45, 0.55, 0.9, 0.1, 0.9, 0.5, 0.9])
    returns = np.asarray([0.0, 0.01, -0.01, 0.02, -0.02, 0.01, 0.0, 0.01])
    config = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    report = threshold_stability_report(
        [(timestamps, returns, probabilities)],
        config,
        timeframe="1h",
    )
    assert [point["delta"] for point in report["points"]] == [-0.04, -0.02, 0.0, 0.02, 0.04]
    assert report["fold_count"] == 1
    assert report["summary"]["all_points_solvable"] is True
    assert report["interpretation"] == "diagnostic_only_not_a_promotion_gate"
    assert (config.short_threshold, config.long_threshold) == (0.42, 0.58)


def test_threshold_stability_rejects_malformed_sweep() -> None:
    timestamps = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    fold = [(timestamps, np.zeros(2), np.asarray([0.4, 0.6]))]
    with pytest.raises(ValueError, match="include"):
        threshold_stability_report(fold, BacktestConfig(), timeframe="1h", deltas=(0.02,))
    with pytest.raises(ValueError, match="unique"):
        threshold_stability_report(fold, BacktestConfig(), timeframe="1h", deltas=(0.0, 0.0))
