import pandas as pd

from ares_engine.data.quality import validate_cross_venue, validate_ohlcv
from ares_engine.synthetic import make_synthetic_ohlcv


def test_valid_ohlcv_passes() -> None:
    frame = make_synthetic_ohlcv(200)
    report = validate_ohlcv(frame, "1h", source="test", max_gap_count=0)
    assert report.passed
    assert report.stats["rows"] == 200


def test_bad_ohlcv_is_rejected() -> None:
    frame = make_synthetic_ohlcv(20)
    frame = frame.drop(index=5).reset_index(drop=True)
    frame.loc[2, "high"] = frame.loc[2, "low"] - 1
    frame = frame._append(frame.iloc[3], ignore_index=True)
    report = validate_ohlcv(frame, "1h", source="test", max_gap_count=0)
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "missing_candles" in codes
    assert "ohlc_invariant" in codes
    assert "duplicate_timestamp" in codes


def test_cross_venue_check_detects_persistent_divergence() -> None:
    primary = make_synthetic_ohlcv(200, seed=5)
    secondary = primary.copy()
    secondary["close"] *= 1.02
    report = validate_cross_venue(
        primary,
        secondary,
        primary_name="a",
        secondary_name="b",
        max_p95_bps=50,
        min_overlap=100,
    )
    assert not report.passed
    assert report.stats["p95_divergence_bps"] > 100


def test_regular_but_off_grid_timestamps_are_rejected() -> None:
    frame = make_synthetic_ohlcv(50)
    frame["timestamp"] = frame["timestamp"] + pd.Timedelta(minutes=5)
    report = validate_ohlcv(frame, "1h", source="test", max_gap_count=0)
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "misaligned_timestamp" in codes
    assert report.stats["off_grid_timestamps"] == len(frame)


def test_latest_cross_venue_outlier_is_rejected_even_when_p95_passes() -> None:
    primary = make_synthetic_ohlcv(400, seed=6)
    secondary = primary.copy()
    secondary.loc[secondary.index[-1], "close"] *= 1.05
    report = validate_cross_venue(
        primary,
        secondary,
        primary_name="a",
        secondary_name="b",
        max_p95_bps=100.0,
        min_overlap=100,
        max_latest_bps=150.0,
    )
    codes = {issue.code for issue in report.issues}
    assert report.stats["p95_divergence_bps"] < 100.0
    assert report.stats["latest_divergence_bps"] > 150.0
    assert "latest_cross_venue_divergence" in codes


def test_latest_candle_must_align_across_venues() -> None:
    primary = make_synthetic_ohlcv(200, seed=7)
    secondary = primary.iloc[:-1].copy()
    report = validate_cross_venue(
        primary,
        secondary,
        primary_name="a",
        secondary_name="b",
        max_p95_bps=50.0,
        min_overlap=100,
        max_latest_bps=100.0,
    )
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "latest_timestamp_misalignment" in codes
    assert report.stats["primary_latest_timestamp"] > report.stats["secondary_latest_timestamp"]


def test_requested_historical_range_must_be_fully_covered() -> None:
    frame = make_synthetic_ohlcv(100, seed=8)
    start = pd.Timestamp(frame["timestamp"].min()) - pd.Timedelta(hours=2)
    end = pd.Timestamp(frame["timestamp"].max()) + pd.Timedelta(hours=3)
    report = validate_ohlcv(
        frame,
        "1h",
        source="test",
        max_gap_count=0,
        expected_start=start,
        expected_end=end,
    )
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "incomplete_start_coverage" in codes
    assert "incomplete_end_coverage" in codes
