"""Optuna search over labels, sequence models, and decision thresholds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, cast

import optuna
import pandas as pd

from .config import AresConfig, dump_config
from .exceptions import AresError
from .utils import atomic_write_json
from .validation import run_walk_forward

STUDY_IDENTITY_SCHEMA = "ares-study-identity-v2"
SEARCH_SPACE = {
    "label_method": ["k_ahead", "triple_barrier"],
    "horizon_bars": [3, 6, 12, 24],
    "dead_zone_bps": {"low": 5.0, "high": 40.0, "step": 5.0},
    "take_profit_bps": {"low": 30.0, "high": 120.0, "step": 10.0},
    "stop_loss_bps": {"low": 20.0, "high": 100.0, "step": 10.0},
    "model_family": ["lstm", "tcn"],
    "lookback_bars": [24, 36, 48, 72, 96, 144],
    "hidden_units": [16, 32, 64, 96, 128],
    "dropout": {"low": 0.0, "high": 0.5, "step": 0.1},
    "learning_rate": {"low": 1e-4, "high": 3e-3, "log": True},
    "tcn_blocks": {"low": 2, "high": 4},
    "tcn_kernel_size": [2, 3, 5],
    "long_threshold": {"low": 0.52, "high": 0.70, "step": 0.02},
    "short_threshold": {"low": 0.30, "high": 0.48, "step": 0.02},
}


def _trial_config(base: AresConfig, trial: optuna.trial.BaseTrial) -> AresConfig:
    config = base.model_copy(deep=True)
    config.labels.method = cast(
        Literal["k_ahead", "triple_barrier"],
        trial.suggest_categorical("label_method", ["k_ahead", "triple_barrier"]),
    )
    config.labels.horizon_bars = trial.suggest_categorical("horizon_bars", [3, 6, 12, 24])
    if config.labels.method == "k_ahead":
        config.labels.dead_zone_bps = trial.suggest_float("dead_zone_bps", 5.0, 40.0, step=5.0)
    else:
        config.labels.take_profit_bps = trial.suggest_float(
            "take_profit_bps", 30.0, 120.0, step=10.0
        )
        config.labels.stop_loss_bps = trial.suggest_float("stop_loss_bps", 20.0, 100.0, step=10.0)

    config.model.family = cast(
        Literal["lstm", "tcn"], trial.suggest_categorical("model_family", ["lstm", "tcn"])
    )
    config.model.lookback_bars = trial.suggest_categorical(
        "lookback_bars", [24, 36, 48, 72, 96, 144]
    )
    config.model.hidden_units = trial.suggest_categorical("hidden_units", [16, 32, 64, 96, 128])
    config.model.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    config.model.learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    if config.model.family == "tcn":
        config.model.tcn_blocks = trial.suggest_int("tcn_blocks", 2, 4)
        config.model.tcn_kernel_size = trial.suggest_categorical("tcn_kernel_size", [2, 3, 5])

    config.backtest.long_threshold = trial.suggest_float("long_threshold", 0.52, 0.70, step=0.02)
    config.backtest.short_threshold = trial.suggest_float("short_threshold", 0.30, 0.48, step=0.02)
    config.validation.purge_bars = config.labels.horizon_bars
    return AresConfig.model_validate(config.model_dump(mode="python"))


def _data_fingerprint(ohlcv: pd.DataFrame, base_config: AresConfig) -> str:
    """Canonical identity for every input that can materially alter a study."""
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required).difference(ohlcv.columns)
    if missing:
        raise ValueError(f"Cannot identify study; OHLCV columns are missing: {sorted(missing)}")

    canonical = ohlcv.copy()
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True, errors="raise")
    value_columns = [*required, *sorted(set(canonical.columns).difference(required))]
    records: list[list[str | int | None]] = []
    for row in canonical[value_columns].itertuples(index=False, name=None):
        record: list[str | int | None] = []
        for column, value in zip(value_columns, row, strict=True):
            if column == "timestamp":
                record.append(int(pd.Timestamp(value).value))
            elif column in {"open", "high", "low", "close", "volume"}:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"Cannot identify study with non-finite {column}")
                record.append(numeric.hex())
            elif pd.isna(value):
                record.append(None)
            else:
                record.append(str(value))
        records.append(record)
    records.sort(key=lambda item: json.dumps(item, separators=(",", ":"), ensure_ascii=True))
    content = json.dumps(
        {"columns": value_columns, "records": records},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    config_identity = {
        "project_seed": base_config.project.seed,
        "data": {
            "primary_exchange": base_config.data.primary_exchange,
            "validation_exchanges": base_config.data.validation_exchanges,
            "symbol": base_config.data.symbol,
            "timeframe": base_config.data.timeframe,
        },
        "features": base_config.features.model_dump(mode="json"),
        "labels": base_config.labels.model_dump(mode="json"),
        "sequences_and_models": base_config.model.model_dump(mode="json"),
        "folds_and_scaling": base_config.validation.model_dump(mode="json"),
        "backtest": base_config.backtest.model_dump(mode="json"),
        "gates": base_config.gates.model_dump(mode="json"),
        "search_space": SEARCH_SPACE,
    }
    payload = json.dumps(
        {
            "schema": STUDY_IDENTITY_SCHEMA,
            "full_normalized_ohlcv_sha256": content_hash,
            "configuration": config_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_search(
    ohlcv: pd.DataFrame,
    base_config: AresConfig,
    *,
    output_dir: Path | None = None,
    verbose: int = 0,
) -> tuple[optuna.Study, AresConfig]:
    output_dir = output_dir or base_config.storage.artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        config = _trial_config(base_config, trial)
        try:
            summary = run_walk_forward(ohlcv, config, verbose=verbose)
        except (AresError, ValueError, RuntimeError) as exc:
            trial.set_user_attr("failure", str(exc))
            trial.set_user_attr("passed", False)
            return -2_000_000.0
        trial.set_user_attr("passed", summary.passed)
        trial.set_user_attr("aggregate", summary.aggregate)
        trial.set_user_attr("gates", summary.gates)
        trial.set_user_attr("raw_score", summary.score)
        if not summary.passed:
            failed_count = sum(not passed for passed in summary.gates.values())
            return -1_000_000.0 - failed_count
        return summary.score

    sampler = optuna.samplers.TPESampler(seed=base_config.project.seed)
    # Trials are only comparable against the exact dataset they were scored on.
    # Scoping the study name by a dataset fingerprint makes resuming on identical
    # data work while making silent cross-vintage selection structurally
    # impossible: a changed dataset always yields a fresh study.
    fingerprint = _data_fingerprint(ohlcv, base_config)
    scoped_study_name = f"{base_config.search.study_name}-{fingerprint[:12]}"
    study = optuna.create_study(
        study_name=scoped_study_name,
        storage=base_config.search.storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    study.set_user_attr("ares_data_fingerprint", fingerprint)
    study.set_user_attr("ares_base_study_name", base_config.search.study_name)
    study.optimize(
        objective,
        n_trials=base_config.search.n_trials,
        timeout=base_config.search.timeout_seconds,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    passing_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and bool(trial.user_attrs.get("passed", False))
        and trial.value is not None
        # NaN or infinite objective values must never rank, let alone win.
        and math.isfinite(float(trial.value))
    ]
    if not passing_trials:
        atomic_write_json(
            output_dir / "best_meta.json",
            {
                "study_name": study.study_name,
                "status": "no_gate_passing_trial",
                "trial_count": len(study.trials),
            },
        )
        raise ValueError("Optuna completed without a gate-passing candidate")

    selected_trial = max(passing_trials, key=lambda trial: float(trial.value or 0.0))
    study.set_user_attr("ares_selected_trial_number", selected_trial.number)
    study.set_user_attr("ares_selected_trial_value", selected_trial.value)
    best_config = _trial_config(base_config, selected_trial)
    atomic_write_json(output_dir / "best_params.json", selected_trial.params)
    atomic_write_json(
        output_dir / "best_meta.json",
        {
            "study_name": study.study_name,
            "best_trial": selected_trial.number,
            "best_value": selected_trial.value,
            "user_attrs": selected_trial.user_attrs,
            "trial_count": len(study.trials),
        },
    )
    dump_config(best_config, output_dir / "best_config.yaml")
    return study, best_config
