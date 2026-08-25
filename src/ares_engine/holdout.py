"""One-shot, post-search historical holdout with reproducible baselines.

The final chronological partition never enters search, scaler fitting, early
stopping, or configuration selection. A persistent commitment consumes its
single evaluation before model inference begins; failed evaluations are never
silently retried.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout
from sklearn.metrics import roc_auc_score

from .backtest import run_backtest
from .config import AresConfig
from .data.quality import validate_ohlcv
from .dataset import fit_scaler, transform_sequences
from .models import build_model, clear_session, fit_model, predict_probabilities
from .search import SEARCH_SPACE, _data_fingerprint, run_search
from .utils import atomic_write_json
from .validation import prepare_dataset

COMMITMENT_SCHEMA = "ares-locked-final-holdout-commitment.v1"
REPORT_SCHEMA = "ares-locked-final-holdout-report.v1"
_TERMINAL_STATES = {"EVALUATION_STARTED", "EVALUATION_COMPLETE", "EVALUATION_FAILED"}


class HoldoutIntegrityError(ValueError):
    """A holdout cannot be evaluated without violating its locked protocol."""


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configuration_hash(config: AresConfig) -> str:
    return _canonical_hash(config.model_dump(mode="json"))


def _check_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise HoldoutIntegrityError("holdout_non_finite_metric")


def _partitions(
    frame: pd.DataFrame,
    config: AresConfig,
    *,
    holdout_bars: int,
    embargo_bars: int | None,
) -> tuple[pd.DataFrame, int, int, int]:
    if isinstance(holdout_bars, bool) or holdout_bars < 50:
        raise HoldoutIntegrityError("holdout_requires_at_least_50_bars")
    max_search_horizon = max(cast(list[int], SEARCH_SPACE["horizon_bars"]))
    minimum_embargo = max(max_search_horizon, config.labels.horizon_bars)
    embargo = minimum_embargo if embargo_bars is None else embargo_bars
    if isinstance(embargo, bool) or embargo < minimum_embargo:
        raise HoldoutIntegrityError("holdout_embargo_shorter_than_maximum_search_horizon")
    boundary = len(frame) - holdout_bars
    research_end = boundary - embargo
    if research_end < config.validation.min_train_bars:
        raise HoldoutIntegrityError("holdout_research_partition_too_small")

    quality = validate_ohlcv(
        frame,
        config.data.timeframe,
        source="locked final holdout input",
        max_gap_count=config.data.max_gap_count,
    )
    if not quality.passed:
        raise HoldoutIntegrityError("holdout_input_failed_data_quality")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise HoldoutIntegrityError("holdout_timestamps_not_strictly_chronological")
    return frame.iloc[:research_end].copy(deep=True), research_end, boundary, embargo


def _benchmark_metrics(
    *,
    timestamps: pd.DatetimeIndex,
    returns: np.ndarray,
    probabilities: np.ndarray,
    config: AresConfig,
) -> dict[str, Any]:
    baseline = run_backtest(
        timestamps,
        returns,
        probabilities,
        config.backtest,
        timeframe=config.data.timeframe,
    )
    stressed = run_backtest(
        timestamps,
        returns,
        probabilities,
        config.backtest,
        timeframe=config.data.timeframe,
        cost_multiplier=config.validation.stress_cost_multiplier,
    )
    result = {
        "backtest": baseline.metrics.to_dict(),
        "stress_backtest": stressed.metrics.to_dict(),
        "solvent": not baseline.metrics.bankrupt and not stressed.metrics.bankrupt,
    }
    _check_finite(result)
    return result


def _evaluate_holdout(
    frame: pd.DataFrame,
    config: AresConfig,
    *,
    research_end: int,
    holdout_start: int,
    verbose: int,
) -> dict[str, Any]:
    _, dataset = prepare_dataset(frame, config)
    horizon = config.labels.horizon_bars
    research = np.flatnonzero(dataset.source_rows + horizon < research_end)
    holdout = np.flatnonzero(
        (dataset.source_rows >= holdout_start) & (dataset.source_rows + horizon < len(frame))
    )
    if len(holdout) < 30:
        raise HoldoutIntegrityError("holdout_has_insufficient_complete_samples")
    validation_size = config.validation.validation_bars
    if len(research) <= validation_size + horizon:
        raise HoldoutIntegrityError("holdout_early_stopping_partition_too_small")
    early_stop = research[-validation_size:]
    training = research[: -(validation_size + horizon)]
    directional = dataset.directional_mask
    training = training[directional[training]]
    early_stop = early_stop[directional[early_stop]]
    if len(training) < 50 or len(np.unique(dataset.y_binary[training])) < 2:
        raise HoldoutIntegrityError("holdout_training_partition_not_directionally_diverse")
    if len(early_stop) < 2 or len(np.unique(dataset.y_binary[early_stop])) < 2:
        raise HoldoutIntegrityError("holdout_early_stopping_partition_not_directionally_diverse")

    scaler = fit_scaler(dataset.X[training])
    train_features = transform_sequences(scaler, dataset.X[training])
    validation_features = transform_sequences(scaler, dataset.X[early_stop])
    holdout_features = transform_sequences(scaler, dataset.X[holdout])
    model = build_model(dataset.X.shape[1:], config.model, seed=config.project.seed)
    try:
        fit_model(
            model,
            train_features,
            dataset.y_binary[training].astype("float32"),
            config.model,
            X_validation=validation_features,
            y_validation=dataset.y_binary[early_stop].astype("float32"),
            verbose=verbose,
        )
        probabilities = predict_probabilities(model, holdout_features)
    finally:
        clear_session()

    timestamps = dataset.timestamps[holdout]
    returns = dataset.bar_returns[holdout]
    candidate = _benchmark_metrics(
        timestamps=timestamps,
        returns=returns,
        probabilities=probabilities,
        config=config,
    )
    directional_holdout = directional[holdout]
    if (
        directional_holdout.any()
        and len(np.unique(dataset.y_binary[holdout][directional_holdout])) > 1
    ):
        candidate["auc"] = float(
            roc_auc_score(
                dataset.y_binary[holdout][directional_holdout],
                probabilities[directional_holdout],
            )
        )
    else:
        candidate["auc"] = None
    benchmarks = {
        "flat_cash": np.full(len(holdout), 0.5, dtype="float64"),
        "buy_and_hold": np.ones(len(holdout), dtype="float64"),
        "lagged_momentum": np.where(returns >= 0.0, 1.0, 0.0),
    }
    baseline_results = {
        name: _benchmark_metrics(
            timestamps=timestamps,
            returns=returns,
            probabilities=values,
            config=config,
        )
        for name, values in benchmarks.items()
    }
    return {
        "sample_count": len(holdout),
        "training_sample_count": len(training),
        "early_stopping_sample_count": len(early_stop),
        "first_timestamp": timestamps[0].isoformat(),
        "last_timestamp": timestamps[-1].isoformat(),
        "candidate": candidate,
        "benchmarks": baseline_results,
        "candidate_beats_buy_and_hold": bool(
            candidate["backtest"]["total_return"]
            > baseline_results["buy_and_hold"]["backtest"]["total_return"]
        ),
    }


def run_locked_holdout(
    frame: pd.DataFrame,
    base_config: AresConfig,
    *,
    holdout_bars: int,
    embargo_bars: int | None = None,
    output_dir: Path | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Run search on quarantined history, then consume one immutable final evaluation."""

    destination = output_dir or (base_config.storage.artifacts / "locked-holdout")
    destination.mkdir(parents=True, exist_ok=True)
    commitment_path = destination / "commitment.json"
    report_path = destination / "final_report.json"
    lock = FileLock(str(destination / ".holdout.lock"), timeout=0)
    try:
        with lock:
            if commitment_path.exists():
                try:
                    existing = json.loads(commitment_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise HoldoutIntegrityError("holdout_commitment_unreadable") from None
                if not isinstance(existing, dict) or existing.get("schema") != COMMITMENT_SCHEMA:
                    raise HoldoutIntegrityError("holdout_commitment_invalid")
                if existing.get("state") in _TERMINAL_STATES:
                    raise HoldoutIntegrityError("holdout_already_consumed")
                raise HoldoutIntegrityError("holdout_existing_commitment_requires_manual_review")

            snapshot = frame.copy(deep=True)
            frozen_config = base_config.model_copy(deep=True)
            research, research_end, holdout_start, embargo = _partitions(
                snapshot,
                frozen_config,
                holdout_bars=holdout_bars,
                embargo_bars=embargo_bars,
            )
            base_hash = _configuration_hash(frozen_config)
            full_fingerprint = _data_fingerprint(snapshot, frozen_config)
            research_fingerprint = _data_fingerprint(research, frozen_config)
            holdout_fingerprint = _data_fingerprint(snapshot.iloc[holdout_start:], frozen_config)
            commitment = {
                "schema": COMMITMENT_SCHEMA,
                "state": "SEARCH_IN_PROGRESS",
                "full_dataset_sha256": full_fingerprint,
                "research_dataset_sha256": research_fingerprint,
                "holdout_dataset_sha256": holdout_fingerprint,
                "base_configuration_sha256": base_hash,
                "research_bars": research_end,
                "embargo_bars": embargo,
                "holdout_bars": holdout_bars,
                "holdout_start_timestamp": pd.Timestamp(
                    snapshot.iloc[holdout_start]["timestamp"]
                ).isoformat(),
                "one_time_evaluation": True,
                "order_execution_enabled": False,
            }
            commitment["commitment_sha256"] = _canonical_hash(
                {key: value for key, value in commitment.items() if key != "state"}
            )
            atomic_write_json(commitment_path, commitment)

            study, selected_config = run_search(
                research.copy(deep=True),
                frozen_config,
                output_dir=destination / "search",
                verbose=verbose,
            )
            if _configuration_hash(frozen_config) != base_hash:
                raise HoldoutIntegrityError("holdout_base_configuration_mutated_during_search")
            if study.user_attrs.get("ares_data_fingerprint") != research_fingerprint:
                raise HoldoutIntegrityError("holdout_search_partition_fingerprint_mismatch")
            if "ares_selected_trial_number" not in study.user_attrs:
                raise HoldoutIntegrityError("holdout_search_winner_not_frozen")
            if selected_config.labels.horizon_bars > embargo:
                raise HoldoutIntegrityError("holdout_selected_horizon_exceeds_embargo")
            if _data_fingerprint(frame, frozen_config) != full_fingerprint:
                raise HoldoutIntegrityError("holdout_dataset_mutated_during_search")
            selected_hash = _configuration_hash(selected_config)

            # Consume the only permitted inspection before inference begins.
            # A crash or failed model evaluation cannot reset this state.
            commitment.update(
                state="EVALUATION_STARTED",
                selected_trial_number=int(study.user_attrs["ares_selected_trial_number"]),
                selected_configuration_sha256=selected_hash,
            )
            atomic_write_json(commitment_path, commitment)
            try:
                result = _evaluate_holdout(
                    snapshot,
                    selected_config,
                    research_end=research_end,
                    holdout_start=holdout_start,
                    verbose=verbose,
                )
                if _configuration_hash(selected_config) != selected_hash:
                    raise HoldoutIntegrityError(
                        "holdout_selected_configuration_mutated_during_evaluation"
                    )
                if _data_fingerprint(frame, frozen_config) != full_fingerprint:
                    raise HoldoutIntegrityError("holdout_dataset_mutated_during_evaluation")
                _check_finite(result)
                if not result["candidate"]["solvent"]:
                    raise HoldoutIntegrityError("holdout_candidate_insolvent")
            except Exception:
                commitment["state"] = "EVALUATION_FAILED"
                atomic_write_json(commitment_path, commitment)
                raise

            report = {
                "schema": REPORT_SCHEMA,
                "commitment_sha256": commitment["commitment_sha256"],
                "full_dataset_sha256": full_fingerprint,
                "research_dataset_sha256": research_fingerprint,
                "holdout_dataset_sha256": holdout_fingerprint,
                "base_configuration_sha256": base_hash,
                "selected_configuration_sha256": selected_hash,
                "selected_trial_number": commitment["selected_trial_number"],
                "research_bars": research_end,
                "embargo_bars": embargo,
                "holdout_bars": holdout_bars,
                "result": result,
                "one_time_evaluation": True,
                "historical_result_only": True,
                "profitability_established": False,
                "order_execution_enabled": False,
            }
            report["report_sha256"] = _canonical_hash(report)
            atomic_write_json(report_path, report)
            commitment["state"] = "EVALUATION_COMPLETE"
            commitment["report_sha256"] = report["report_sha256"]
            atomic_write_json(commitment_path, commitment)
            return report
    except Timeout:
        raise HoldoutIntegrityError("holdout_evaluation_already_running") from None
