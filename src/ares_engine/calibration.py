"""Research-fold reliability and cost-aware threshold diagnostics, never tuning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestMetrics, run_backtest
from .config import BacktestConfig

WALK_FORWARD_PROTOCOL = "ares-walk-forward-train-loss-v1"


def evidence_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_digest(value: np.ndarray) -> str:
    """Canonical little-endian float64 array identity, including shape."""
    values = np.ascontiguousarray(value, dtype="<f8")
    # Canonicalize missing labels and signed zero before hashing.
    values = values.copy()
    values[np.isnan(values)] = np.nan
    values[values == 0] = 0.0
    digest = hashlib.sha256(json.dumps(list(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    values = np.asarray(value, dtype="float64")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return values


def probability_calibration_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bin_count: int = 10,
    reference_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize binary probability reliability without fitting on evaluation data."""
    labels = _vector(labels, "labels")
    probabilities = _vector(probabilities, "probabilities")
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same non-zero length")
    if type(bin_count) is not int or not 2 <= bin_count <= 1000:
        raise ValueError("bin_count must be an integer between 2 and 1000")
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
    brier = float(np.mean((probabilities - labels) ** 2))
    baselines: dict[str, Any] = {"constant_half_brier": 0.25}
    if reference_probabilities is not None:
        reference = _vector(reference_probabilities, "reference probabilities")
        if len(reference) != len(labels) or ((reference < 0) | (reference > 1)).any():
            raise ValueError("reference probabilities must match labels and lie within [0, 1]")
        reference_brier = float(np.mean((reference - labels) ** 2))
        skill = 1 - brier / reference_brier if reference_brier > 0 else None
        if skill is not None and not np.isfinite(skill):
            skill = None
        baselines.update(
            training_prior_brier=reference_brier,
            brier_skill_vs_training_prior=skill,
            skill_unavailable_reason="zero_or_numerically_unstable_reference_brier"
            if skill is None
            else None,
        )
    return {
        "schema": "ares-probability-calibration-v2",
        "sample_count": len(labels),
        "positive_count": int(labels.sum()),
        "bin_count": bin_count,
        "brier_score": brier,
        "baselines": baselines,
        "class_count": int(len(np.unique(labels))),
        "limitations": ["descriptive_only_no_confidence_interval"]
        + (["single_class_evaluation"] if len(np.unique(labels)) < 2 else [])
        + (["fewer_samples_than_bins"] if len(labels) < bin_count else []),
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
    stress_cost_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Describe fixed predictions, including incomplete perturbation coverage."""
    config = BacktestConfig.model_validate(config.model_dump(mode="python"))
    fold_inputs = list(folds)
    if not fold_inputs:
        raise ValueError("threshold stability needs at least one fold")
    if not deltas or any(
        type(delta) not in (int, float) or not np.isfinite(delta) for delta in deltas
    ):
        raise ValueError("deltas must be finite numbers")
    if 0.0 not in deltas or len(set(deltas)) != len(deltas):
        raise ValueError("deltas must be unique and include the configured threshold")
    if not np.isfinite(stress_cost_multiplier) or stress_cost_multiplier < 1:
        raise ValueError("stress cost multiplier must be finite and at least one")
    fold_evidence = []
    timestamp_sets = []
    for timestamps, returns, probabilities in fold_inputs:
        returns = _vector(returns, "returns")
        probabilities = _vector(probabilities, "probabilities")
        if (
            not isinstance(timestamps, pd.DatetimeIndex)
            or timestamps.tz is None
            or timestamps.hasnans
            or not timestamps.is_monotonic_increasing
            or timestamps.has_duplicates
        ):
            raise ValueError("fold timestamps must be timezone-aware, sorted, unique and valid")
        if len(timestamps) <= config.execution_delay_bars:
            raise ValueError("fold has insufficient bars for an executed observation")
        if len(timestamps) != len(returns) or len(returns) != len(probabilities):
            raise ValueError("fold inputs must have identical lengths")
        if ((probabilities < 0) | (probabilities > 1)).any() or (returns < -1).any():
            raise ValueError("probabilities or returns outside their supported range")
        timestamp_values = [int(stamp.value) for stamp in timestamps]
        timestamp_sets.append(set(timestamp_values))
        fold_evidence.append(
            {
                "sample_count": len(timestamps),
                "first_timestamp": timestamps[0].isoformat(),
                "last_timestamp": timestamps[-1].isoformat(),
                "timestamps_sha256": evidence_digest(timestamp_values),
                "returns_sha256": array_digest(returns),
                "probabilities_sha256": array_digest(probabilities),
            }
        )

    def metrics_for(
        perturbed: BacktestConfig, multiplier: float, baseline: str | None = None
    ) -> list[BacktestMetrics]:
        results = []
        for timestamps, returns, probabilities in fold_inputs:
            if baseline is not None:
                probabilities = np.full(
                    len(timestamps),
                    {"flat_cash": 0.5, "buy_and_hold": 1.0, "always_short": 0.0}[baseline],
                )
            metrics = run_backtest(
                timestamps,
                returns,
                probabilities,
                perturbed,
                timeframe=timeframe,
                cost_multiplier=multiplier,
            ).metrics
            if any(
                isinstance(value, float) and not np.isfinite(value)
                for value in metrics.to_dict().values()
            ):
                raise ValueError("non-finite backtest metrics cannot support threshold evidence")
            results.append(metrics)
        return results

    def summarize(metrics: list[BacktestMetrics]) -> dict[str, Any]:
        result = {
            "median_return": float(np.median([item.total_return for item in metrics])),
            "median_sharpe": float(np.median([item.sharpe for item in metrics])),
            "worst_drawdown": float(max(item.max_drawdown for item in metrics)),
            "median_turnover": float(np.median([item.turnover for item in metrics])),
            "total_trades": int(sum(item.trades for item in metrics)),
            "bankrupt_folds": int(sum(item.bankrupt for item in metrics)),
        }
        if any(isinstance(value, float) and not np.isfinite(value) for value in result.values()):
            raise ValueError("non-finite aggregate threshold evidence")
        return result

    points: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_pairs: set[tuple[float, float]] = set()
    # Reserve the exact base pair before processing nearby floating-point deltas.
    base_pair = (config.long_threshold, config.short_threshold)
    for delta in sorted(deltas):
        long_threshold = config.long_threshold + delta
        short_threshold = config.short_threshold - delta
        if not (0.5 < long_threshold < 1.0 and 0.0 < short_threshold < 0.5):
            skipped.append({"delta": delta, "reason": "outside_supported_decision_range"})
            continue
        pair = (long_threshold, short_threshold)
        if pair in seen_pairs or (delta != 0 and pair == base_pair):
            skipped.append({"delta": delta, "reason": "duplicate_effective_thresholds"})
            continue
        seen_pairs.add(pair)
        perturbed = BacktestConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
            }
        )
        metrics = metrics_for(perturbed, 1.0)
        points.append(
            {
                "delta": delta,
                "long_threshold": long_threshold,
                "short_threshold": short_threshold,
                **summarize(metrics),
                "stress": summarize(metrics_for(perturbed, stress_cost_multiplier)),
                "fold_metrics": [item.to_dict() for item in metrics],
            }
        )
    point_returns = [float(point["median_return"]) for point in points]
    sharpes = [float(point["median_sharpe"]) for point in points]
    result = {
        "schema": "ares-threshold-stability-v2",
        "fold_count": len(fold_inputs),
        "points": points,
        "requested_deltas": list(deltas),
        "skipped_points": skipped,
        "coverage": "complete" if not skipped else "partial",
        "sensitivity_available": len(points) > 1,
        "fold_evidence": fold_evidence,
        "observation_count": sum(len(stamps) for stamps in timestamp_sets),
        "unique_timestamp_count": len(set().union(*timestamp_sets)),
        "aggregation": "fold_medians_with_repeated_timestamps_counted_per_fold",
        "assumptions": {
            **config.model_dump(mode="json"),
            "timeframe": timeframe,
            "stress_cost_multiplier": stress_cost_multiplier,
        },
        "benchmarks": {
            name: {
                "base": summarize(metrics_for(config, 1.0, name)),
                "stress": summarize(metrics_for(config, stress_cost_multiplier, name)),
            }
            for name in ("flat_cash", "buy_and_hold", "always_short")
        },
        "summary": {
            "median_return_range": max(point_returns) - min(point_returns),
            "median_sharpe_range": max(sharpes) - min(sharpes),
            "positive_return_fraction": sum(value > 0.0 for value in point_returns)
            / len(point_returns),
            "all_points_solvable": all(
                int(point["bankrupt_folds"]) == 0 and int(point["stress"]["bankrupt_folds"]) == 0
                for point in points
            ),
            "all_evaluated_points_solvent_at_base_cost": all(
                int(point["bankrupt_folds"]) == 0 for point in points
            ),
            "all_evaluated_points_solvent_under_stress": all(
                int(point["stress"]["bankrupt_folds"]) == 0 for point in points
            ),
        },
        "interpretation": "diagnostic_only_not_a_promotion_gate",
    }
    # Strict serialization also rejects arithmetic overflow in ranges.
    try:
        evidence_digest(result)
    except ValueError as exc:
        raise ValueError("non-finite threshold report") from exc
    return result
