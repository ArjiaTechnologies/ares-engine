"""Walk-forward model validation, realistic backtesting, and rejection gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .backtest import BacktestMetrics, run_backtest
from .calibration import (
    WALK_FORWARD_PROTOCOL,
    array_digest,
    evidence_digest,
    probability_calibration_report,
    threshold_stability_report,
)
from .config import AresConfig
from .data.quality import validate_ohlcv
from .dataset import (
    PurgedWalkForwardSplitter,
    SequenceDataset,
    build_sequence_dataset,
    fit_scaler,
    transform_sequences,
)
from .exceptions import DataQualityError
from .features import FeatureFrame, build_features
from .labels import build_labels
from .models import build_model, clear_session, fit_model, predict_probabilities


@dataclass(slots=True)
class FoldMetrics:
    fold: int
    train_samples: int
    validation_samples: int
    train_directional_samples: int
    validation_directional_samples: int
    auc: float | None
    backtest: BacktestMetrics
    stress_backtest: BacktestMetrics

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backtest"] = self.backtest.to_dict()
        payload["stress_backtest"] = self.stress_backtest.to_dict()
        return payload


@dataclass(slots=True)
class ValidationSummary:
    folds: list[FoldMetrics]
    aggregate: dict[str, float | int | bool]
    gates: dict[str, bool]
    passed: bool
    score: float
    feature_columns: list[str]
    sample_count: int
    probability_calibration: dict[str, Any]
    threshold_stability: dict[str, Any]
    diagnostic_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "aggregate": self.aggregate,
            "gates": self.gates,
            "passed": self.passed,
            "score": self.score,
            "feature_columns": self.feature_columns,
            "sample_count": self.sample_count,
            "probability_calibration": self.probability_calibration,
            "threshold_stability": self.threshold_stability,
            "diagnostic_provenance": self.diagnostic_provenance,
        }


def feature_warmup_rows(config: AresConfig) -> int:
    """Index of the first row whose full feature vector is finite (exact arithmetic)."""
    features = config.features
    candidates = [
        max(features.ema_periods) - 1,
        features.rsi_period,
        features.bollinger_period - 1,
        max(features.volatility_windows),
        features.volume_z_window - 1,
        1,
    ]
    return max(candidates)


def minimum_required_bars(config: AresConfig) -> int:
    """Smallest OHLCV row count that can satisfy the configured walk-forward split."""
    required_samples = (
        config.validation.min_train_bars
        + int(config.validation.purge_bars or 0)
        + config.validation.validation_bars
    )
    return required_samples + feature_warmup_rows(config) + config.model.lookback_bars - 1


def prepare_dataset(
    ohlcv: pd.DataFrame, config: AresConfig
) -> tuple[FeatureFrame, SequenceDataset]:
    features = build_features(ohlcv, config.features)
    labels = build_labels(features.frame, config.labels)
    dataset = build_sequence_dataset(
        features.frame,
        features.columns,
        labels.labels,
        lookback_bars=config.model.lookback_bars,
    )
    return features, dataset


def _auc_or_none(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _aggregate(folds: list[FoldMetrics]) -> dict[str, float | int | bool]:
    if not folds:
        raise ValueError("Walk-forward validation produced no folds")
    sharpe = np.asarray([fold.backtest.sharpe for fold in folds], dtype="float64")
    returns = np.asarray([fold.backtest.total_return for fold in folds], dtype="float64")
    drawdowns = np.asarray([fold.backtest.max_drawdown for fold in folds], dtype="float64")
    turnover = np.asarray([fold.backtest.turnover for fold in folds], dtype="float64")
    stress_returns = np.asarray(
        [fold.stress_backtest.total_return for fold in folds], dtype="float64"
    )
    aucs = np.asarray([fold.auc for fold in folds if fold.auc is not None], dtype="float64")
    bankrupt_folds = sum(fold.backtest.bankrupt or fold.stress_backtest.bankrupt for fold in folds)
    return {
        "fold_count": len(folds),
        "median_sharpe": float(np.median(sharpe)),
        "mean_sharpe": float(np.mean(sharpe)),
        "sharpe_std": float(np.std(sharpe)),
        "median_return": float(np.median(returns)),
        "mean_return": float(np.mean(returns)),
        "worst_drawdown": float(np.max(drawdowns)),
        "median_turnover": float(np.median(turnover)),
        "total_trades": int(sum(fold.backtest.trades for fold in folds)),
        "mean_auc": float(np.mean(aucs)) if len(aucs) else 0.5,
        "median_stress_return": float(np.median(stress_returns)),
        "all_stress_positive": bool(np.all(stress_returns > 0.0)),
        "bankrupt_folds": bankrupt_folds,
    }


def robust_score(aggregate: dict[str, float | int | bool]) -> float:
    """Reward risk-adjusted consistency; punish drawdown and split instability."""
    return float(
        float(aggregate["median_sharpe"])
        + 2.0 * float(aggregate["median_return"])
        + 0.5 * (float(aggregate["mean_auc"]) - 0.5)
        - 1.5 * float(aggregate["worst_drawdown"])
        - 0.25 * float(aggregate["sharpe_std"])
    )


def evaluate_gates(aggregate: dict[str, float | int | bool], config: AresConfig) -> dict[str, bool]:
    gates = {
        "median_sharpe": float(aggregate["median_sharpe"]) >= config.gates.min_median_sharpe,
        "median_return": float(aggregate["median_return"]) >= config.gates.min_median_return,
        "drawdown": float(aggregate["worst_drawdown"]) <= config.gates.max_worst_drawdown,
        "turnover": float(aggregate["median_turnover"]) <= config.gates.max_median_turnover,
        "trade_count": int(aggregate["total_trades"]) >= config.gates.min_total_trades,
        "solvency": int(aggregate["bankrupt_folds"]) == 0,
    }
    if config.gates.require_positive_cost_stress:
        gates["cost_stress"] = bool(aggregate["all_stress_positive"])
    return gates


def run_walk_forward(
    ohlcv: pd.DataFrame,
    config: AresConfig,
    *,
    verbose: int = 0,
) -> ValidationSummary:
    # Revalidate mutable configuration before partitioning or model construction.
    config = AresConfig.model_validate(config.model_dump(mode="python"))
    # Hash and evaluate the same owned snapshot, even during a long model fit.
    ohlcv = ohlcv.copy(deep=True)
    data_report = validate_ohlcv(
        ohlcv,
        config.data.timeframe,
        source="walk-forward input",
        max_gap_count=config.data.max_gap_count,
    )
    if not data_report.passed:
        messages = "; ".join(issue.message for issue in data_report.issues)
        raise DataQualityError(f"Walk-forward input failed data gates: {messages}")
    _, dataset = prepare_dataset(ohlcv, config)
    splitter = PurgedWalkForwardSplitter(
        min_train_size=config.validation.min_train_bars,
        validation_size=config.validation.validation_bars,
        step_size=config.validation.step_bars,
        purge_size=int(config.validation.purge_bars or 0),
        max_splits=config.validation.max_folds,
    )
    minimum_samples = (
        config.validation.min_train_bars
        + int(config.validation.purge_bars or 0)
        + config.validation.validation_bars
    )
    if len(dataset.X) < minimum_samples:
        raise ValueError(
            f"Walk-forward validation needs at least {minimum_samples} complete sequence samples; "
            f"only {len(dataset.X)} are available"
        )

    folds: list[FoldMetrics] = []
    calibration_labels: list[np.ndarray] = []
    calibration_probabilities: list[np.ndarray] = []
    calibration_references: list[np.ndarray] = []
    calibration_indices: list[np.ndarray] = []
    partition_evidence: list[dict[str, Any]] = []
    threshold_folds: list[tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]] = []
    directional = dataset.directional_mask
    y_binary = dataset.y_binary

    for fold in splitter.split(len(dataset.X)):
        validation_rows = dataset.source_rows[fold.validation_indices]
        train_rows = dataset.source_rows[fold.train_indices]
        if int(train_rows[-1]) + config.labels.horizon_bars >= int(validation_rows[0]):
            raise ValueError("training label horizon overlaps evaluation partition")
        train_directional = fold.train_indices[directional[fold.train_indices]]
        validation_directional = fold.validation_indices[directional[fold.validation_indices]]
        if len(train_directional) < 50 or len(np.unique(y_binary[train_directional])) < 2:
            raise ValueError(
                f"Fold {fold.number} has insufficient directional training data or only one class"
            )

        scaler = fit_scaler(dataset.X[fold.train_indices])
        X_train_directional = transform_sequences(scaler, dataset.X[train_directional])
        X_validation_all = transform_sequences(scaler, dataset.X[fold.validation_indices])
        y_train = y_binary[train_directional].astype("float32")

        model = build_model(
            dataset.X.shape[1:], config.model, seed=config.project.seed + fold.number
        )
        try:
            # fit_model uses training loss when no validation_data is supplied.
            # Outer research outcomes must never select this fold's weights.
            fit_model(model, X_train_directional, y_train, config.model, verbose=verbose)
            probabilities = predict_probabilities(model, X_validation_all)
        finally:
            clear_session()
        train_prior = float(y_train.mean())
        partition_evidence.append(
            {
                "fold": fold.number,
                "train_sample_count": len(fold.train_indices),
                "train_directional_count": len(train_directional),
                "evaluation_sample_count": len(fold.validation_indices),
                "evaluation_directional_count": len(validation_directional),
                "evaluation_neutral_count": int(
                    (dataset.labels[fold.validation_indices] == 0).sum()
                ),
                "evaluation_missing_label_count": int(
                    (~np.isfinite(dataset.labels[fold.validation_indices])).sum()
                ),
                "training_positive_rate": train_prior,
                "last_training_label_source_row": int(train_rows[-1]) + config.labels.horizon_bars,
                "first_evaluation_source_row": int(validation_rows[0]),
                "last_requested_evaluation_label_source_row": int(validation_rows[-1])
                + config.labels.horizon_bars,
                "train_source_rows_sha256": evidence_digest(train_rows.tolist()),
                "evaluation_source_rows_sha256": evidence_digest(validation_rows.tolist()),
                "train_features_sha256": array_digest(dataset.X[fold.train_indices]),
                "training_labels_sha256": array_digest(y_train),
                "evaluation_features_sha256": array_digest(dataset.X[fold.validation_indices]),
                "evaluation_labels_sha256": array_digest(dataset.labels[fold.validation_indices]),
                "directional_source_rows_sha256": evidence_digest(
                    dataset.source_rows[validation_directional].tolist()
                ),
                "predictions_sha256": array_digest(probabilities),
            }
        )
        threshold_folds.append(
            (
                dataset.timestamps[fold.validation_indices],
                dataset.bar_returns[fold.validation_indices],
                probabilities,
            )
        )
        auc = None
        if len(validation_directional):
            validation_positions = np.searchsorted(fold.validation_indices, validation_directional)
            calibration_labels.append(y_binary[validation_directional])
            calibration_probabilities.append(probabilities[validation_positions])
            calibration_references.append(np.full(len(validation_directional), train_prior))
            calibration_indices.append(validation_directional)
            auc = _auc_or_none(
                y_binary[validation_directional],
                probabilities[validation_positions],
            )

        base = run_backtest(
            dataset.timestamps[fold.validation_indices],
            dataset.bar_returns[fold.validation_indices],
            probabilities,
            config.backtest,
            timeframe=config.data.timeframe,
        )
        stress = run_backtest(
            dataset.timestamps[fold.validation_indices],
            dataset.bar_returns[fold.validation_indices],
            probabilities,
            config.backtest,
            timeframe=config.data.timeframe,
            cost_multiplier=config.validation.stress_cost_multiplier,
        )
        folds.append(
            FoldMetrics(
                fold=fold.number,
                train_samples=len(fold.train_indices),
                validation_samples=len(fold.validation_indices),
                train_directional_samples=len(train_directional),
                validation_directional_samples=len(validation_directional),
                auc=auc,
                backtest=base.metrics,
                stress_backtest=stress.metrics,
            )
        )

    aggregate = _aggregate(folds)
    gates = evaluate_gates(aggregate, config)
    if not calibration_labels:
        raise ValueError("Walk-forward validation produced no directional calibration samples")
    calibration = probability_calibration_report(
        np.concatenate(calibration_labels),
        np.concatenate(calibration_probabilities),
        reference_probabilities=np.concatenate(calibration_references),
    )
    unique_samples = len(np.unique(np.concatenate(calibration_indices)))
    calibration.update(
        population="directional_labels_only",
        unique_sample_count=unique_samples,
        repeated_observation_count=calibration["sample_count"] - unique_samples,
        aggregation="pooled_fold_observations_repeats_counted",
    )
    source_columns = ["open", "high", "low", "close", "volume"]
    provenance = {
        "schema": "ares-diagnostic-provenance-v1",
        "fit_protocol": WALK_FORWARD_PROTOCOL,
        "early_stopping_monitor": "training_loss",
        "evaluation_used_for_fit": False,
        "scope": "research_only_optuna_can_select_on_these_folds",
        "configuration_sha256": evidence_digest(config.model_dump(mode="json")),
        "source_columns": ["timestamp", *source_columns],
        "source_sha256": evidence_digest(
            {
                "timestamps_ns": [
                    int(value.value) for value in pd.to_datetime(ohlcv["timestamp"], utc=True)
                ],
                "ohlcv_sha256": array_digest(ohlcv[source_columns].to_numpy(dtype="float64")),
            }
        ),
        "label_horizon_bars": config.labels.horizon_bars,
        "purge_bars": config.validation.purge_bars,
        "partitions": partition_evidence,
        "limitations": [
            "not_independent_forward_evidence",
            "no_calibrator_fitted",
            "no_threshold_selected",
            "overlapping_folds_are_not_independent",
        ],
    }
    return ValidationSummary(
        folds=folds,
        aggregate=aggregate,
        gates=gates,
        passed=all(gates.values()),
        score=robust_score(aggregate),
        feature_columns=dataset.feature_columns,
        sample_count=len(dataset.X),
        probability_calibration=calibration,
        threshold_stability=threshold_stability_report(
            threshold_folds,
            config.backtest,
            timeframe=config.data.timeframe,
            stress_cost_multiplier=config.validation.stress_cost_multiplier,
        ),
        diagnostic_provenance=provenance,
    )
