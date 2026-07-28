"""Hard market-data sanity checks used before training or paper inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils import timeframe_to_seconds

Severity = Literal["warning", "error"]


@dataclass(slots=True)
class QualityIssue:
    code: str
    severity: Severity
    message: str
    count: int = 1


@dataclass(slots=True)
class QualityReport:
    source: str
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
            "stats": self.stats,
        }


def _issue(issues: list[QualityIssue], code: str, message: str, count: int = 1) -> None:
    issues.append(QualityIssue(code=code, severity="error", message=message, count=int(count)))


def validate_ohlcv(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    source: str,
    max_gap_count: int = 0,
    expected_exchange: str | None = None,
    expected_symbol: str | None = None,
    expected_timeframe: str | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    max_staleness_bars: int | None = None,
    expected_start: datetime | pd.Timestamp | None = None,
    expected_end: datetime | pd.Timestamp | None = None,
) -> QualityReport:
    """Validate schema, metadata, chronology, continuity, freshness, and OHLC invariants."""
    issues: list[QualityIssue] = []
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        _issue(
            issues, "missing_columns", f"Missing required OHLCV columns: {missing}", len(missing)
        )
        return QualityReport(source=source, passed=False, issues=issues, stats={"rows": len(frame)})
    if frame.empty:
        _issue(issues, "empty_dataset", "No OHLCV rows were returned")
        return QualityReport(source=source, passed=False, issues=issues, stats={"rows": 0})

    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    invalid_timestamps = int(work["timestamp"].isna().sum())
    if invalid_timestamps:
        _issue(issues, "invalid_timestamp", "Rows contain invalid timestamps", invalid_timestamps)

    duplicate_count = int(work["timestamp"].duplicated().sum())
    if duplicate_count:
        _issue(
            issues, "duplicate_timestamp", "Duplicate candle timestamps detected", duplicate_count
        )

    if not work["timestamp"].is_monotonic_increasing:
        _issue(issues, "unsorted_timestamp", "Candle timestamps are not monotonically increasing")

    expected_metadata = {
        "exchange": expected_exchange,
        "symbol": expected_symbol,
        "timeframe": expected_timeframe,
    }
    for column, expected_value in expected_metadata.items():
        if expected_value is None:
            continue
        if column not in work:
            _issue(
                issues,
                "missing_metadata",
                f"{source} is missing required {column} metadata",
            )
            continue
        observed = {str(value) for value in work[column].dropna().unique()}
        if observed != {expected_value}:
            _issue(
                issues,
                "metadata_mismatch",
                f"{source} {column} metadata {sorted(observed)} does not match {expected_value}",
            )

    numeric = work[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype="float64"))).sum())
    if non_finite:
        _issue(
            issues, "non_finite_value", "OHLCV contains NaN or infinite numeric values", non_finite
        )

    non_positive_prices = int((numeric[["open", "high", "low", "close"]] <= 0).sum().sum())
    if non_positive_prices:
        _issue(
            issues,
            "non_positive_price",
            "OHLC prices must be strictly positive",
            non_positive_prices,
        )

    negative_volume = int((numeric["volume"] < 0).sum())
    if negative_volume:
        _issue(issues, "negative_volume", "Volume must be non-negative", negative_volume)

    high_invalid = numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
    low_invalid = numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    invariant_count = int((high_invalid | low_invalid).sum())
    if invariant_count:
        _issue(
            issues,
            "ohlc_invariant",
            "High/low bounds are inconsistent with open/close",
            invariant_count,
        )

    expected = pd.Timedelta(timeframe_to_seconds(timeframe), unit="s")
    expected_ns = int(expected.value)
    valid_timestamps = work["timestamp"].dropna().sort_values()
    delta_ns = valid_timestamps.diff().dropna().astype("int64")
    # Integer ceiling division catches a missing expected slot even when a malformed timestamp
    # lands between grid points, without float rounding on nanosecond values.
    gap_steps = ((delta_ns + expected_ns - 1) // expected_ns - 1).clip(lower=0)
    gap_count = int(gap_steps.sum())
    irregular_count = int((delta_ns.mod(expected_ns) != 0).sum())
    off_grid_count = int(((valid_timestamps.astype("int64") % expected_ns) != 0).sum())
    if gap_count > max_gap_count:
        _issue(
            issues,
            "missing_candles",
            f"Detected {gap_count} missing candle slots; allowed maximum is {max_gap_count}",
            gap_count,
        )
    if irregular_count:
        _issue(
            issues,
            "irregular_interval",
            "Consecutive candle timestamps are not exact multiples of the configured frequency",
            irregular_count,
        )
    if off_grid_count:
        _issue(
            issues,
            "misaligned_timestamp",
            "Candle timestamps do not sit on the configured UTC frequency grid",
            off_grid_count,
        )

    expected_first_timestamp: pd.Timestamp | None = None
    expected_last_timestamp: pd.Timestamp | None = None
    if expected_start is not None:
        start_reference = pd.Timestamp(expected_start)
        start_reference = (
            start_reference.tz_localize(UTC)
            if start_reference.tzinfo is None
            else start_reference.tz_convert(UTC)
        )
        expected_first_ns = ((start_reference.value + expected_ns - 1) // expected_ns) * expected_ns
        expected_first_timestamp = pd.Timestamp(expected_first_ns, unit="ns", tz="UTC")
        if not valid_timestamps.empty and valid_timestamps.min() > expected_first_timestamp:
            missing_initial = int(
                np.ceil((valid_timestamps.min() - expected_first_timestamp) / expected)
            )
            _issue(
                issues,
                "incomplete_start_coverage",
                f"{source} starts at {valid_timestamps.min()}, after required first candle "
                f"{expected_first_timestamp}",
                max(1, missing_initial),
            )
    if expected_end is not None:
        end_reference = pd.Timestamp(expected_end)
        end_reference = (
            end_reference.tz_localize(UTC)
            if end_reference.tzinfo is None
            else end_reference.tz_convert(UTC)
        )
        expected_last_ns = ((end_reference.value - 1) // expected_ns) * expected_ns
        expected_last_timestamp = pd.Timestamp(expected_last_ns, unit="ns", tz="UTC")
        if not valid_timestamps.empty and valid_timestamps.max() < expected_last_timestamp:
            missing_terminal = int(
                np.ceil((expected_last_timestamp - valid_timestamps.max()) / expected)
            )
            _issue(
                issues,
                "incomplete_end_coverage",
                f"{source} ends at {valid_timestamps.max()}, before required last candle "
                f"{expected_last_timestamp}",
                max(1, missing_terminal),
            )

    staleness_bars: float | None = None
    if max_staleness_bars is not None:
        if max_staleness_bars < 0:
            raise ValueError("max_staleness_bars must be non-negative")
        if valid_timestamps.empty:
            staleness_bars = float("inf")
        else:
            reference = pd.Timestamp(as_of or datetime.now(tz=UTC))
            reference = (
                reference.tz_localize(UTC)
                if reference.tzinfo is None
                else reference.tz_convert(UTC)
            )
            latest_close = valid_timestamps.max() + expected
            stale_seconds = max(0.0, (reference - latest_close).total_seconds())
            staleness_bars = float(stale_seconds / expected.total_seconds())
        if staleness_bars > max_staleness_bars:
            _issue(
                issues,
                "stale_data",
                f"{source} data is {staleness_bars:.2f} bars stale; maximum is "
                f"{max_staleness_bars}",
            )

    stats = {
        "rows": int(len(work)),
        "start": work["timestamp"].min(),
        "end": work["timestamp"].max(),
        "duplicates": duplicate_count,
        "missing_candle_slots": gap_count,
        "irregular_intervals": irregular_count,
        "off_grid_timestamps": off_grid_count,
        "close_min": float(numeric["close"].min()),
        "close_max": float(numeric["close"].max()),
        "volume_total": float(numeric["volume"].sum()),
    }
    if staleness_bars is not None:
        stats["staleness_bars"] = staleness_bars
    if expected_first_timestamp is not None:
        stats["expected_first_timestamp"] = expected_first_timestamp
    if expected_last_timestamp is not None:
        stats["expected_last_timestamp"] = expected_last_timestamp
    return QualityReport(source=source, passed=not issues, issues=issues, stats=stats)


def validate_cross_venue(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    primary_name: str,
    secondary_name: str,
    max_p95_bps: float,
    min_overlap: int,
    max_latest_bps: float | None = None,
) -> QualityReport:
    """Compare aligned closes across venues and reject large persistent disagreement."""
    required = {"timestamp", "close"}
    missing_primary = sorted(required.difference(primary.columns))
    missing_secondary = sorted(required.difference(secondary.columns))
    if missing_primary or missing_secondary:
        column_issues: list[QualityIssue] = []
        if missing_primary:
            _issue(
                column_issues,
                "missing_cross_venue_columns",
                f"{primary_name} is missing cross-venue columns: {missing_primary}",
                len(missing_primary),
            )
        if missing_secondary:
            _issue(
                column_issues,
                "missing_cross_venue_columns",
                f"{secondary_name} is missing cross-venue columns: {missing_secondary}",
                len(missing_secondary),
            )
        return QualityReport(
            source=f"{primary_name}~{secondary_name}",
            passed=False,
            issues=column_issues,
            stats={"overlap_rows": 0},
        )

    left = primary[["timestamp", "close"]].rename(columns={"close": "primary_close"}).copy()
    right = secondary[["timestamp", "close"]].rename(columns={"close": "secondary_close"}).copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
    left["primary_close"] = pd.to_numeric(left["primary_close"], errors="coerce")
    right["secondary_close"] = pd.to_numeric(right["secondary_close"], errors="coerce")
    left = left.replace([np.inf, -np.inf], np.nan).dropna()
    right = right.replace([np.inf, -np.inf], np.nan).dropna()
    left = left.loc[left["primary_close"] > 0]
    right = right.loc[right["secondary_close"] > 0]
    joined = left.merge(right, on="timestamp", how="inner").dropna()

    issues: list[QualityIssue] = []
    if len(joined) < min_overlap:
        _issue(
            issues,
            "insufficient_overlap",
            f"Only {len(joined)} aligned candles between {primary_name} and {secondary_name}; "
            f"minimum is {min_overlap}",
            max(1, min_overlap - len(joined)),
        )

    if joined.empty:
        return QualityReport(
            source=f"{primary_name}~{secondary_name}",
            passed=False,
            issues=issues,
            stats={"overlap_rows": 0},
        )

    midpoint = (joined["primary_close"] + joined["secondary_close"]) / 2.0
    divergence_bps = (
        (joined["primary_close"] - joined["secondary_close"]).abs() / midpoint
    ) * 10_000
    p50 = float(divergence_bps.quantile(0.50))
    p95 = float(divergence_bps.quantile(0.95))
    maximum = float(divergence_bps.max())
    latest_index = joined["timestamp"].idxmax()
    latest_timestamp = pd.Timestamp(joined["timestamp"].loc[latest_index])
    primary_latest_timestamp = pd.Timestamp(left["timestamp"].max())
    secondary_latest_timestamp = pd.Timestamp(right["timestamp"].max())
    latest_divergence = float(divergence_bps.loc[latest_index])
    if (
        latest_timestamp != primary_latest_timestamp
        or latest_timestamp != secondary_latest_timestamp
    ):
        _issue(
            issues,
            "latest_timestamp_misalignment",
            "Newest candle is not aligned across venues: "
            f"{primary_name}={primary_latest_timestamp}, "
            f"{secondary_name}={secondary_latest_timestamp}, "
            f"latest overlap={latest_timestamp}",
        )
    if p95 > max_p95_bps:
        _issue(
            issues,
            "cross_venue_divergence",
            f"95th percentile close divergence is {p95:.2f} bps, above {max_p95_bps:.2f} bps",
        )
    if max_latest_bps is not None and latest_divergence > max_latest_bps:
        _issue(
            issues,
            "latest_cross_venue_divergence",
            f"Latest aligned close divergence is {latest_divergence:.2f} bps, "
            f"above {max_latest_bps:.2f} bps",
        )

    stats = {
        "overlap_rows": int(len(joined)),
        "median_divergence_bps": p50,
        "p95_divergence_bps": p95,
        "max_divergence_bps": maximum,
        "latest_divergence_bps": latest_divergence,
        "latest_aligned_timestamp": latest_timestamp,
        "primary_latest_timestamp": primary_latest_timestamp,
        "secondary_latest_timestamp": secondary_latest_timestamp,
        "start": joined["timestamp"].min(),
        "end": joined["timestamp"].max(),
    }
    return QualityReport(
        source=f"{primary_name}~{secondary_name}",
        passed=not issues,
        issues=issues,
        stats=stats,
    )
