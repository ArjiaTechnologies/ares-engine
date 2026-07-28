"""Feature and label construction verified against independent hand-loop references."""

import math

import numpy as np
import pandas as pd
import pytest
from reference_impl import (
    ref_bollinger,
    ref_ema,
    ref_k_ahead,
    ref_log_returns,
    ref_realized_vol,
    ref_triple_barrier,
    ref_volume_z,
    ref_wilder_rsi,
)

from ares_engine.config import FeatureConfig, LabelConfig
from ares_engine.features import build_features
from ares_engine.labels import build_labels
from ares_engine.synthetic import make_synthetic_ohlcv


def _assert_series_matches(series: pd.Series, reference: list, name: str) -> None:
    values = series.to_numpy(dtype="float64")
    assert len(values) == len(reference)
    for index, expected in enumerate(reference):
        actual = values[index]
        if expected is None:
            assert math.isnan(actual), f"{name}[{index}] expected NaN, got {actual}"
        else:
            assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12), f"{name}[{index}]"


@pytest.fixture()
def config() -> FeatureConfig:
    return FeatureConfig(
        ema_periods=[3, 5],
        rsi_period=4,
        bollinger_period=4,
        bollinger_std=2.0,
        volatility_windows=[3],
        volume_z_window=4,
    )


@pytest.fixture()
def frame() -> pd.DataFrame:
    ohlcv = make_synthetic_ohlcv(60, seed=99)
    return ohlcv


def test_ema_matches_reference(frame, config) -> None:
    built = build_features(frame, config).frame
    closes = frame["close"].tolist()
    for period in [3, 5]:
        _assert_series_matches(built[f"ema_{period}"], ref_ema(closes, period), f"ema_{period}")
        expected_distance = [
            None if e is None else c / e - 1.0
            for c, e in zip(closes, ref_ema(closes, period), strict=True)
        ]
        _assert_series_matches(
            built[f"close_to_ema_{period}"], expected_distance, f"close_to_ema_{period}"
        )


def test_rsi_matches_reference_on_mixed_series(frame, config) -> None:
    built = build_features(frame, config).frame
    _assert_series_matches(built["rsi_4"], ref_wilder_rsi(frame["close"].tolist(), 4), "rsi_4")


def test_rsi_flat_series_is_50_up_is_100_down_is_0() -> None:
    base = make_synthetic_ohlcv(30, seed=1)
    config = FeatureConfig(
        ema_periods=[3], rsi_period=4, bollinger_period=4, volatility_windows=[3], volume_z_window=4
    )
    flat = base.copy()
    flat["close"] = 100.0
    up = base.copy()
    up["close"] = np.linspace(100, 130, len(base))
    down = base.copy()
    down["close"] = np.linspace(130, 100, len(base))
    assert build_features(flat, config).frame["rsi_4"].dropna().eq(50.0).all()
    assert build_features(up, config).frame["rsi_4"].dropna().eq(100.0).all()
    assert build_features(down, config).frame["rsi_4"].dropna().eq(0.0).all()


def test_bollinger_matches_reference(frame, config) -> None:
    built = build_features(frame, config).frame
    closes = frame["close"].tolist()
    mids, highs, lows, positions = ref_bollinger(closes, 4, 2.0)
    _assert_series_matches(built["bb_mid_4"], mids, "bb_mid")
    _assert_series_matches(built["bb_high_4"], highs, "bb_high")
    _assert_series_matches(built["bb_low_4"], lows, "bb_low")
    _assert_series_matches(built["bb_position_4"], positions, "bb_position")


def test_bollinger_zero_width_yields_nan_position_not_infinity() -> None:
    base = make_synthetic_ohlcv(20, seed=2)
    base["close"] = 250.0
    config = FeatureConfig(
        ema_periods=[3], rsi_period=4, bollinger_period=4, volatility_windows=[3], volume_z_window=4
    )
    built = build_features(base, config).frame
    tail = built["bb_position_4"].iloc[4:]
    assert tail.isna().all()


def test_log_returns_and_realized_vol_match_reference(frame, config) -> None:
    built = build_features(frame, config).frame
    closes = frame["close"].tolist()
    log_returns = ref_log_returns(closes)
    _assert_series_matches(built["log_return_1"], log_returns, "log_return_1")
    _assert_series_matches(
        built["realized_vol_3"], ref_realized_vol(log_returns, 3), "realized_vol_3"
    )


def test_volume_features_match_reference(frame, config) -> None:
    built = build_features(frame, config).frame
    volumes = frame["volume"].tolist()
    expected_log = [math.log1p(v) for v in volumes]
    _assert_series_matches(built["log_volume"], expected_log, "log_volume")
    _assert_series_matches(built["volume_z_4"], ref_volume_z(volumes, 4), "volume_z_4")


def test_range_features_match_reference(frame, config) -> None:
    built = build_features(frame, config).frame
    expected_range = [
        (h - low) / c for h, low, c in zip(frame["high"], frame["low"], frame["close"], strict=True)
    ]
    expected_open_close = [c / o - 1.0 for c, o in zip(frame["close"], frame["open"], strict=True)]
    _assert_series_matches(built["high_low_range"], expected_range, "high_low_range")
    _assert_series_matches(built["close_open_return"], expected_open_close, "close_open_return")


def test_feature_column_order_is_deterministic_and_complete(frame, config) -> None:
    first = build_features(frame, config)
    second = build_features(frame.sample(frac=1.0, random_state=3), config)
    assert first.columns == second.columns
    expected = [
        "close",
        "ema_3",
        "close_to_ema_3",
        "ema_5",
        "close_to_ema_5",
        "rsi_4",
        "bb_mid_4",
        "bb_high_4",
        "bb_low_4",
        "bb_position_4",
        "log_return_1",
        "realized_vol_3",
        "log_volume",
        "volume_z_4",
        "high_low_range",
        "close_open_return",
    ]
    assert first.columns == expected
    np.testing.assert_allclose(
        first.frame[first.columns].to_numpy(),
        second.frame[second.columns].to_numpy(),
        equal_nan=True,
        err_msg="shuffled input must produce identical sorted feature values",
    )


def test_warmup_rows_are_nan_and_first_complete_row_is_exact(frame, config) -> None:
    built = build_features(frame, config)
    matrix = built.frame[built.columns]
    complete = matrix.notna().all(axis=1)
    # Deepest requirement: volume_z window 4 -> index 3; rsi 4 -> index 4; vol_3 over diff -> index 3.
    first_complete = int(complete.idxmax())
    assert first_complete == 4
    assert not complete.iloc[:4].any()
    assert complete.iloc[4:].all()


def test_missing_input_column_raises(frame, config) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_features(frame.drop(columns=["high"]), config)


def test_k_ahead_labels_match_reference_and_horizon_is_exact() -> None:
    frame = make_synthetic_ohlcv(50, seed=5)
    label_config = LabelConfig(method="k_ahead", horizon_bars=7, dead_zone_bps=25.0)
    result = build_labels(frame, label_config)
    reference = ref_k_ahead(frame["close"].tolist(), 7, 25.0)
    for index, expected in enumerate(reference):
        actual = result.labels.iloc[index]
        if expected is None:
            assert math.isnan(actual)
        else:
            assert actual == expected
    assert result.labels.iloc[-7:].isna().all()
    assert not math.isnan(result.labels.iloc[-8])
    assert list(result.labels.index) == list(frame.index)


def test_k_ahead_dead_zone_boundaries_are_strict() -> None:
    frame = make_synthetic_ohlcv(10, seed=6)
    frame["close"] = 10_000.0
    frame.loc[frame.index[3], "close"] = 10_000.0 * (
        1 + 0.0015
    )  # exactly +dz for row 2? use explicit rows
    # Row 0 horizon lands on row 1..; craft explicit: horizon 1, dz 15bps
    closes = [10_000.0, 10_000.0 * 1.00150001, 10_000.0, 10_000.0 * 0.99849999, 10_000.0, 10_000.0]
    frame = frame.iloc[: len(closes)].copy()
    frame["close"] = closes
    labels = build_labels(
        frame, LabelConfig(method="k_ahead", horizon_bars=1, dead_zone_bps=15.0)
    ).labels
    assert labels.iloc[0] == 1.0  # just above dead zone
    assert labels.iloc[2] == -1.0  # just below negative dead zone
    zero_dz = build_labels(
        frame, LabelConfig(method="k_ahead", horizon_bars=1, dead_zone_bps=0.0)
    ).labels
    assert zero_dz.iloc[4] == 0.0  # exactly zero return with zero dead zone stays neutral


def test_triple_barrier_matches_reference_on_random_data() -> None:
    frame = make_synthetic_ohlcv(300, seed=8)
    config = LabelConfig(
        method="triple_barrier", horizon_bars=6, take_profit_bps=80, stop_loss_bps=60
    )
    result = build_labels(frame, config)
    reference = ref_triple_barrier(
        frame["close"].tolist(), frame["high"].tolist(), frame["low"].tolist(), 6, 80, 60
    )
    for index, expected in enumerate(reference):
        actual = result.labels.iloc[index]
        if expected is None:
            assert math.isnan(actual)
        else:
            assert actual == expected, f"row {index}"


def _flat_frame(closes: list[float]) -> pd.DataFrame:
    frame = make_synthetic_ohlcv(len(closes), seed=9)
    frame["open"] = closes
    frame["close"] = closes
    frame["high"] = closes
    frame["low"] = closes
    return frame


def test_triple_barrier_exact_touch_counts_as_hit() -> None:
    closes = [10_000.0] * 6
    frame = _flat_frame(closes)
    config = LabelConfig(
        method="triple_barrier", horizon_bars=3, take_profit_bps=100, stop_loss_bps=100
    )
    frame.loc[frame.index[2], "high"] = 10_100.0  # exactly the profit barrier of row 0
    labels = build_labels(frame, config).labels
    assert labels.iloc[0] == 1.0
    assert labels.iloc[1] == 1.0  # barrier for row 1 is also exactly touched at row 2


def test_triple_barrier_current_candle_cannot_trigger_its_own_barrier() -> None:
    closes = [10_000.0] * 6
    frame = _flat_frame(closes)
    config = LabelConfig(
        method="triple_barrier", horizon_bars=2, take_profit_bps=50, stop_loss_bps=50
    )
    # Row 2's own high is far above its barrier, but scanning must start at row 3.
    frame.loc[frame.index[2], "high"] = 11_000.0
    labels = build_labels(frame, config).labels
    assert labels.iloc[2] == 0.0  # expires untouched; its own candle is excluded
    assert labels.iloc[0] == 1.0  # earlier rows see row 2's spike inside their window
    assert labels.iloc[1] == 1.0


def test_triple_barrier_simultaneous_hit_is_neutral_and_stops_scanning() -> None:
    closes = [10_000.0] * 8
    frame = _flat_frame(closes)
    config = LabelConfig(
        method="triple_barrier", horizon_bars=4, take_profit_bps=50, stop_loss_bps=50
    )
    frame.loc[frame.index[2], "high"] = 10_100.0
    frame.loc[frame.index[2], "low"] = 9_900.0
    # Row 3 would be a clean profit hit, but row 2's ambiguity must already have ended row 0/1 scans.
    frame.loc[frame.index[3], "high"] = 10_100.0
    labels = build_labels(frame, config).labels
    assert labels.iloc[0] == 0.0
    assert labels.iloc[1] == 0.0


def test_triple_barrier_expiry_and_tail_nan() -> None:
    closes = [10_000.0] * 7
    frame = _flat_frame(closes)
    config = LabelConfig(
        method="triple_barrier", horizon_bars=3, take_profit_bps=500, stop_loss_bps=500
    )
    result = build_labels(frame, config)
    assert (result.labels.iloc[:4] == 0.0).all()
    assert result.labels.iloc[4:].isna().all()
    assert result.future_return.iloc[0] == pytest.approx(0.0)


def test_labels_are_pure_functions_of_input(config) -> None:
    frame = make_synthetic_ohlcv(120, seed=10)
    for method in ["k_ahead", "triple_barrier"]:
        label_config = LabelConfig(method=method, horizon_bars=5)
        first = build_labels(frame, label_config).labels
        second = build_labels(frame.copy(), label_config).labels
        pd.testing.assert_series_equal(first, second)
