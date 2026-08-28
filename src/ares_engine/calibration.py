"""Out-of-fold probability calibration and decision-threshold diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import BacktestConfig


def probability_calibration_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Summarize binary probability reliability without fitting on evaluation data."""
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same non-zero length")
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0.0) | (probabilities > 1.0)).any()
    ):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    labels = labels.astype("float64")

    # floor maps 1.0 past the last equal-width bin, so clamp it explicitly.
    bin_indices = np.minimum((probabilities * bin_count).astype(int), bin_count - 1)
    bins: list[dict[str, float | int]] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(bin_count):
        mask = bin_indices == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(probabilities[mask].mean())
        positive_rate = float(labels[mask].mean())
        gap = abs(mean_probability - positive_rate)
        weighted_gap += count * gap
        maximum_gap = max(maximum_gap, gap)
        bins.append(
            {
                "lower_bound": index / bin_count,
                "upper_bound": (index + 1) / bin_count,
                "count": count,
                "mean_probability": mean_probability,
                "positive_rate": positive_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "schema": "ares-probability-calibration-v1",
        "sample_count": len(labels),
        "positive_count": int(labels.sum()),
        "bin_count": bin_count,
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "expected_calibration_error": weighted_gap / len(labels),
        "maximum_calibration_error": maximum_gap,
        "bins": bins,
        "interpretation": "diagnostic_only_not_a_promotion_gate",
    }


def threshold_stability_report(
    folds: Iterable[tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]],
    config: BacktestConfig,
    *,
    timeframe: str,
    deltas: tuple[float, ...] = (-0.04, -0.02, 0.0, 0.02, 0.04),
) -> dict[str, Any]:
    """Re-run untouched fold predictions across nearby symmetric thresholds."""
    fold_inputs = list(folds)
    if not fold_inputs:
        raise ValueError("threshold stability needs at least one fold")
    if 0.0 not in deltas or len(set(deltas)) != len(deltas):
        raise ValueError("deltas must be unique and include the configured threshold")

    points: list[dict[str, Any]] = []
    for delta in sorted(deltas):
        long_threshold = config.long_threshold + delta
        short_threshold = config.short_threshold - delta
        if not (0.5 < long_threshold < 1.0 and 0.0 < short_threshold < 0.5):
            raise ValueError("threshold perturbation leaves the supported decision range")
        if short_threshold >= long_threshold:
            raise ValueError("threshold perturbation reverses threshold order")
        perturbed = config.model_copy(
            update={
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
            }
        )
        metrics = [
            run_backtest(timestamps, returns, probabilities, perturbed, timeframe=timeframe).metrics
            for timestamps, returns, probabilities in fold_inputs
        ]
        points.append(
            {
                "delta": delta,
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                "median_return": float(np.median([item.total_return for item in metrics])),
                "median_sharpe": float(np.median([item.sharpe for item in metrics])),
                "worst_drawdown": float(max(item.max_drawdown for item in metrics)),
                "median_turnover": float(np.median([item.turnover for item in metrics])),
                "total_trades": int(sum(item.trades for item in metrics)),
                "bankrupt_folds": int(sum(item.bankrupt for item in metrics)),
            }
        )
    returns = [float(point["median_return"]) for point in points]
    sharpes = [float(point["median_sharpe"]) for point in points]
    return {
        "schema": "ares-threshold-stability-v1",
        "fold_count": len(fold_inputs),
        "points": points,
        "summary": {
            "median_return_range": max(returns) - min(returns),
            "median_sharpe_range": max(sharpes) - min(sharpes),
            "positive_return_fraction": sum(value > 0.0 for value in returns) / len(returns),
            "all_points_solvable": all(int(point["bankrupt_folds"]) == 0 for point in points),
        },
        "interpretation": "diagnostic_only_not_a_promotion_gate",
    }
