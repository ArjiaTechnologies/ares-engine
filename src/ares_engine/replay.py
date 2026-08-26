"""Leakage-resistant recent-window challenger replay evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtest
from .bundles import LoadedBundle, load_bundle, read_verified_bundle_json, verify_bundle
from .config import AresConfig, GateConfig
from .data.quality import validate_ohlcv
from .data.storage import read_market
from .dataset import transform_sequences
from .exceptions import DataQualityError, PromotionRejected
from .models import predict_probabilities
from .search import _data_fingerprint, run_search
from .training import train_candidate
from .utils import atomic_write_json, sha256_file, utc_now
from .validation import minimum_required_bars, prepare_dataset


def split_replay_window(
    frame: pd.DataFrame, config: AresConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the exact latest raw bars before any search or candidate fitting."""
    replay_bars = config.gates.recent_replay_bars
    minimum = minimum_required_bars(config)
    if len(frame) < minimum + replay_bars:
        raise ValueError(
            "Recent-window replay needs at least "
            f"{minimum + replay_bars} bars ({minimum} research + {replay_bars} replay); "
            f"only {len(frame)} are available"
        )
    research = frame.iloc[:-replay_bars].copy(deep=True).reset_index(drop=True)
    replay = frame.iloc[-replay_bars:].copy(deep=True).reset_index(drop=True)
    research_end = pd.Timestamp(research["timestamp"].iloc[-1])
    replay_start = pd.Timestamp(replay["timestamp"].iloc[0])
    if replay_start <= research_end:
        raise ValueError("Recent replay window does not follow the research partition")
    return research, replay


def _timestamp(value: Any, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise PromotionRejected(f"Replay evidence has an invalid {label}") from exc
    if parsed.tzinfo is None:
        raise PromotionRejected(f"Replay evidence {label} must be timezone-aware")
    return parsed.tz_convert("UTC")


def replay_gates(
    *,
    sample_count: int,
    backtest: dict[str, Any],
    stress_backtest: dict[str, Any],
    gates: GateConfig,
) -> dict[str, bool]:
    """Recompute configured replay gates from reportable metrics."""
    for label, payload in (("backtest", backtest), ("stress_backtest", stress_backtest)):
        if not isinstance(payload, dict):
            raise PromotionRejected(f"Replay {label} metrics must be an object")
        for key in ("total_return", "max_drawdown", "turnover", "final_equity"):
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PromotionRejected(f"Replay {label}.{key} must be numeric")
            if not math.isfinite(float(value)):
                raise PromotionRejected(f"Replay {label}.{key} must be finite")
        if not isinstance(payload.get("trades"), int) or isinstance(payload.get("trades"), bool):
            raise PromotionRejected(f"Replay {label}.trades must be an integer")
        if not isinstance(payload.get("bankrupt"), bool):
            raise PromotionRejected(f"Replay {label}.bankrupt must be boolean")

    return {
        "exact_sample_count": sample_count == gates.recent_replay_bars,
        "minimum_trade_count": int(backtest["trades"]) >= gates.min_recent_replay_trades,
        "maximum_drawdown": float(backtest["max_drawdown"]) <= gates.max_recent_replay_drawdown,
        "solvency": not bool(backtest["bankrupt"]) and not bool(stress_backtest["bankrupt"]),
        "cost_stress": (
            float(stress_backtest["total_return"]) > 0.0
            if gates.require_positive_recent_replay_stress
            else True
        ),
    }


def validate_replay_report(
    report: dict[str, Any],
    *,
    challenger: Path,
    gates: GateConfig,
) -> None:
    """Fail closed on stale, cross-bundle, overlapping, or weakened evidence."""
    if report.get("schema_version") != 1:
        raise PromotionRejected("Replay evidence has an unsupported schema version")
    bundle = report.get("bundle")
    source = report.get("source")
    training = report.get("training")
    window = report.get("window")
    if not all(isinstance(value, dict) for value in (bundle, source, training, window)):
        raise PromotionRejected("Replay evidence is missing required object sections")
    assert isinstance(bundle, dict)
    assert isinstance(source, dict)
    assert isinstance(training, dict)
    assert isinstance(window, dict)
    expected_manifest = sha256_file(challenger / "manifest.json")
    if bundle.get("manifest_sha256") != expected_manifest:
        raise PromotionRejected("Replay evidence is bound to a different challenger manifest")
    provenance = read_verified_bundle_json(challenger, "provenance.json")
    source_hash = source.get("sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash.lower())
    ):
        raise PromotionRejected("Replay evidence source hash is invalid")
    normalized_identity = source.get("normalized_dataset_identity_sha256")
    if (
        not isinstance(normalized_identity, str)
        or len(normalized_identity) != 64
        or any(character not in "0123456789abcdef" for character in normalized_identity.lower())
    ):
        raise PromotionRejected("Replay normalized source identity is invalid")
    sample_count = window.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise PromotionRejected("Replay evidence sample count must be an integer")
    trained_through = _timestamp(training.get("latest_feature_timestamp"), label="training cutoff")
    bundled_cutoff = _timestamp(
        provenance.get("latest_feature_timestamp"), label="bundle training cutoff"
    )
    if trained_through != bundled_cutoff:
        raise PromotionRejected("Replay training cutoff disagrees with challenger provenance")
    window_start = _timestamp(window.get("start"), label="window start")
    window_end = _timestamp(window.get("end"), label="window end")
    source_end = _timestamp(source.get("latest_timestamp"), label="source end")
    if not trained_through < window_start <= window_end:
        raise PromotionRejected("Replay window overlaps candidate research history")
    if window_end != source_end:
        raise PromotionRejected("Replay window is not anchored to the latest source bar")
    backtest = report.get("backtest")
    stress_backtest = report.get("stress_backtest")
    if not isinstance(backtest, dict) or not isinstance(stress_backtest, dict):
        raise PromotionRejected("Replay evidence is missing backtest metrics")
    computed = replay_gates(
        sample_count=sample_count,
        backtest=backtest,
        stress_backtest=stress_backtest,
        gates=gates,
    )
    if report.get("gates") != computed or report.get("passed") is not all(computed.values()):
        raise PromotionRejected("Replay evidence gates do not match configured recomputation")
    if not all(computed.values()):
        failed = sorted(name for name, passed in computed.items() if not passed)
        raise PromotionRejected(f"Challenger failed recent replay gates: {failed}")
    if report.get("profitability_established") is not False:
        raise PromotionRejected("Replay evidence must not claim profitability")
    if report.get("order_execution_enabled") is not False:
        raise PromotionRejected("Replay evidence must not claim order execution")


def run_challenger_replay(
    challenger: Path,
    frame: pd.DataFrame,
    *,
    source_path: Path,
    report_path: Path,
    gates: GateConfig,
) -> dict[str, Any]:
    """Replay a verified candidate on the exact latest unseen sequence window."""
    if report_path.exists():
        raise FileExistsError(f"Replay report already exists: {report_path}")
    if not source_path.is_file() or source_path.is_symlink():
        raise PromotionRejected("Replay source must be a regular non-symbolic file")
    verify_bundle(challenger)
    loaded: LoadedBundle = load_bundle(challenger)
    reopened = read_market(source_path)
    supplied_identity = _data_fingerprint(frame, loaded.config)
    reopened_identity = _data_fingerprint(reopened, loaded.config)
    if supplied_identity != reopened_identity:
        raise PromotionRejected("Replay frame does not match the exact source file")
    frame = reopened
    quality = validate_ohlcv(
        frame,
        loaded.config.data.timeframe,
        source="challenger replay input",
        max_gap_count=loaded.config.data.max_gap_count,
    )
    if not quality.passed:
        messages = "; ".join(issue.message for issue in quality.issues)
        raise DataQualityError(f"Challenger replay input failed data gates: {messages}")
    _, dataset = prepare_dataset(frame, loaded.config)
    if dataset.feature_columns != loaded.feature_spec.get("columns"):
        raise PromotionRejected("Replay features do not match the verified challenger bundle")
    trained_through = _timestamp(
        loaded.provenance.get("latest_feature_timestamp"), label="bundle training cutoff"
    )
    eligible_indices = (dataset.timestamps > trained_through).nonzero()[0]
    if len(eligible_indices) < gates.recent_replay_bars:
        raise PromotionRejected(
            "Insufficient unseen recent samples for challenger replay: "
            f"need {gates.recent_replay_bars}, found {len(eligible_indices)}"
        )
    selected = eligible_indices[-gates.recent_replay_bars :]
    if selected[-1] != len(dataset.X) - 1:
        raise PromotionRejected("Replay window does not end at the latest complete sequence")
    probabilities = predict_probabilities(
        loaded.model, transform_sequences(loaded.scaler, dataset.X[selected])
    )
    base = run_backtest(
        dataset.timestamps[selected],
        dataset.bar_returns[selected],
        probabilities,
        loaded.config.backtest,
        timeframe=loaded.config.data.timeframe,
    )
    stress = run_backtest(
        dataset.timestamps[selected],
        dataset.bar_returns[selected],
        probabilities,
        loaded.config.backtest,
        timeframe=loaded.config.data.timeframe,
        cost_multiplier=loaded.config.validation.stress_cost_multiplier,
    )
    base_metrics = base.metrics.to_dict()
    stress_metrics = stress.metrics.to_dict()
    computed_gates = replay_gates(
        sample_count=len(selected),
        backtest=base_metrics,
        stress_backtest=stress_metrics,
        gates=gates,
    )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "bundle": {
            "path": str(challenger),
            "manifest_sha256": sha256_file(challenger / "manifest.json"),
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "normalized_dataset_identity_sha256": reopened_identity,
            "bytes": source_path.stat().st_size,
            "latest_timestamp": dataset.timestamps[selected[-1]],
        },
        "training": {"latest_feature_timestamp": trained_through},
        "window": {
            "start": dataset.timestamps[selected[0]],
            "end": dataset.timestamps[selected[-1]],
            "sample_count": len(selected),
        },
        "backtest": base_metrics,
        "stress_backtest": stress_metrics,
        "gates": computed_gates,
        "passed": all(computed_gates.values()),
        "profitability_established": False,
        "order_execution_enabled": False,
    }
    atomic_write_json(report_path, report)
    validate_replay_report(report, challenger=challenger, gates=gates)
    return report


def prepare_replayed_challenger(
    frame: pd.DataFrame,
    config: AresConfig,
    *,
    source_path: Path,
    repository_root: Path | None = None,
    bundle_name: str | None = None,
    run_search_first: bool = True,
    verbose: int = 0,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Search/train only the earlier partition, then replay the frozen bundle."""
    research, _ = split_replay_window(frame, config)
    candidate_config = config
    if run_search_first:
        _, candidate_config = run_search(research, config)
    bundle, metrics = train_candidate(
        research,
        candidate_config,
        source_identity_sha256=_data_fingerprint(research, candidate_config),
        bundle_name=bundle_name,
        repository_root=repository_root,
        verbose=verbose,
    )
    report_path = candidate_config.storage.artifacts / "replays" / f"{bundle.name}.json"
    report = run_challenger_replay(
        bundle,
        frame,
        source_path=source_path,
        report_path=report_path,
        gates=candidate_config.gates,
    )
    return bundle, metrics, report_path, report
