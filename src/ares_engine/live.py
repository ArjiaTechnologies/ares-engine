"""Read-only champion loading and fail-closed paper signal generation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from .bundles import load_bundle
from .data.quality import QualityReport, validate_cross_venue, validate_ohlcv
from .exceptions import AresError, DataQualityError
from .features import build_features
from .models import predict_probabilities
from .utils import sha256_file, timeframe_to_seconds, utc_now

SecondaryInput: TypeAlias = pd.DataFrame | Mapping[str, pd.DataFrame] | None


@dataclass(slots=True)
class PaperSignal:
    timestamp: pd.Timestamp
    probability: float
    signal: int
    close: float
    bundle: str
    data_quality_passed: bool
    cross_venue_p95_bps: dict[str, float]
    cross_venue_latest_bps: dict[str, float]
    staleness_bars: float
    secondary_staleness_bars: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _report_failures(report: QualityReport) -> list[str]:
    return [f"{report.source}: {issue.message}" for issue in report.issues]


def _open_candle_failures(
    frame: pd.DataFrame,
    *,
    source: str,
    timeframe: str,
    reference: datetime | pd.Timestamp,
) -> list[str]:
    """Reject feeds that still contain the in-progress candle at ``reference``."""
    if "timestamp" not in frame.columns or frame.empty:
        return []
    step = timeframe_to_seconds(timeframe)
    reference_ts = pd.Timestamp(reference)
    reference_ts = (
        reference_ts.tz_localize("UTC")
        if reference_ts.tzinfo is None
        else reference_ts.tz_convert("UTC")
    )
    current_start = pd.Timestamp((int(reference_ts.timestamp()) // step) * step, unit="s", tz="UTC")
    stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    open_rows = int((stamps >= current_start).sum())
    if open_rows:
        return [
            f"{source}: feed contains {open_rows} in-progress or future candle(s) at or after "
            f"{current_start}; only completed candles are allowed for paper inference"
        ]
    return []


def _normalize_secondary_frames(
    secondary_ohlcv: SecondaryInput,
    configured_exchanges: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if secondary_ohlcv is None:
        return {}, []
    if isinstance(secondary_ohlcv, pd.DataFrame):
        if len(configured_exchanges) != 1:
            return {}, [
                "a single secondary DataFrame is only valid when exactly one validation exchange "
                "is configured"
            ]
        return {configured_exchanges[0]: secondary_ohlcv}, []
    frames = {str(exchange): frame for exchange, frame in secondary_ohlcv.items()}
    invalid = [
        exchange for exchange, frame in frames.items() if not isinstance(frame, pd.DataFrame)
    ]
    failures = [f"secondary feed {exchange} is not a pandas DataFrame" for exchange in invalid]
    for exchange in invalid:
        frames.pop(exchange)
    unexpected = sorted(set(frames).difference(configured_exchanges))
    if unexpected:
        failures.append(f"unexpected secondary feeds supplied: {unexpected}")
    return frames, failures


def generate_paper_signal(
    bundle_path: Path,
    primary_ohlcv: pd.DataFrame,
    *,
    secondary_ohlcv: SecondaryInput = None,
    log_path: Path | None = None,
    as_of: datetime | pd.Timestamp | None = None,
) -> PaperSignal:
    """Generate one paper signal only after every configured data gate passes.

    This function fails closed. A stale, malformed, divergent, mismatched, or missing validation feed
    raises ``DataQualityError`` and produces no signal log entry.
    """
    bundle = load_bundle(bundle_path)
    config = bundle.config
    reference = as_of or utc_now()
    failures: list[str] = []
    primary_report = validate_ohlcv(
        primary_ohlcv,
        config.data.timeframe,
        source=config.data.primary_exchange,
        max_gap_count=config.data.max_gap_count,
        expected_exchange=config.data.primary_exchange,
        expected_symbol=config.data.symbol,
        expected_timeframe=config.data.timeframe,
        as_of=reference,
        max_staleness_bars=config.data.max_staleness_bars,
    )
    failures.extend(_report_failures(primary_report))
    if config.data.drop_open_candle:
        failures.extend(
            _open_candle_failures(
                primary_ohlcv,
                source=config.data.primary_exchange,
                timeframe=config.data.timeframe,
                reference=reference,
            )
        )
    staleness_bars = float(primary_report.stats.get("staleness_bars", float("inf")))

    secondary_frames, normalization_failures = _normalize_secondary_frames(
        secondary_ohlcv,
        config.data.validation_exchanges,
    )
    failures.extend(normalization_failures)
    cross_p95: dict[str, float] = {}
    cross_latest: dict[str, float] = {}
    secondary_staleness: dict[str, float] = {}
    required_cross_columns = {"timestamp", "close"}

    for secondary_name in config.data.validation_exchanges:
        secondary = secondary_frames.get(secondary_name)
        if secondary is None:
            failures.append(f"configured secondary validation feed {secondary_name} is missing")
            continue

        secondary_report = validate_ohlcv(
            secondary,
            config.data.timeframe,
            source=secondary_name,
            max_gap_count=config.data.max_gap_count,
            expected_exchange=secondary_name,
            expected_symbol=config.data.symbol,
            expected_timeframe=config.data.timeframe,
            as_of=reference,
            max_staleness_bars=config.data.max_staleness_bars,
        )
        failures.extend(_report_failures(secondary_report))
        if config.data.drop_open_candle:
            failures.extend(
                _open_candle_failures(
                    secondary,
                    source=secondary_name,
                    timeframe=config.data.timeframe,
                    reference=reference,
                )
            )
        stale = float(secondary_report.stats.get("staleness_bars", float("inf")))
        secondary_staleness[secondary_name] = stale

        if required_cross_columns.issubset(
            primary_ohlcv.columns
        ) and required_cross_columns.issubset(secondary.columns):
            cross = validate_cross_venue(
                primary_ohlcv,
                secondary,
                primary_name=config.data.primary_exchange,
                secondary_name=secondary_name,
                max_p95_bps=config.data.max_cross_venue_p95_bps,
                min_overlap=config.data.min_cross_venue_overlap,
                max_latest_bps=config.data.max_cross_venue_latest_bps,
            )
            failures.extend(_report_failures(cross))
            if "p95_divergence_bps" in cross.stats:
                cross_p95[secondary_name] = float(cross.stats["p95_divergence_bps"])
            if "latest_divergence_bps" in cross.stats:
                cross_latest[secondary_name] = float(cross.stats["latest_divergence_bps"])

    if failures:
        raise DataQualityError("Paper inference blocked by data gates: " + "; ".join(failures))

    feature_frame = build_features(primary_ohlcv, config.features)
    expected_columns = list(bundle.feature_spec["columns"])
    if feature_frame.columns != expected_columns:
        raise ValueError(
            "Feature specification mismatch. Refusing inference rather than silently reordering inputs."
        )
    lookback = int(bundle.feature_spec["lookback_bars"])
    inference_window = feature_frame.frame.tail(lookback)
    if len(inference_window) < lookback:
        raise DataQualityError(f"Need at least {lookback} rows for the latest inference window")
    window = inference_window[expected_columns].to_numpy(dtype="float64")
    if not np.isfinite(window).all():
        raise DataQualityError(
            "Latest inference window contains incomplete or non-finite features; refusing to use "
            "an older row"
        )
    scaled_2d = bundle.scaler.transform(window).astype("float32")
    if not np.isfinite(scaled_2d).all():
        raise DataQualityError("Scaled inference window contains non-finite values")
    scaled = scaled_2d[None, :, :]
    probability = float(predict_probabilities(bundle.model, scaled)[0])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise AresError(f"Model emitted an invalid probability: {probability!r}")
    long_threshold = float(bundle.feature_spec["thresholds"]["long"])
    short_threshold = float(bundle.feature_spec["thresholds"]["short"])
    signal = 1 if probability >= long_threshold else -1 if probability <= short_threshold else 0
    latest = inference_window.iloc[-1]
    result = PaperSignal(
        timestamp=pd.Timestamp(latest["timestamp"]),
        probability=probability,
        signal=signal,
        close=float(latest["close"]),
        bundle=str(bundle_path),
        data_quality_passed=True,
        cross_venue_p95_bps=cross_p95,
        cross_venue_latest_bps=cross_latest,
        staleness_bars=staleness_bars,
        secondary_staleness_bars=secondary_staleness,
    )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["recorded_at"] = utc_now().isoformat()
        payload["manifest_sha256"] = sha256_file(bundle_path / "manifest.json")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return result
