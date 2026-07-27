import numpy as np

from ares_engine.config import FeatureConfig, LabelConfig
from ares_engine.features import build_features
from ares_engine.labels import build_labels
from ares_engine.synthetic import make_synthetic_ohlcv


def test_feature_construction_is_backward_looking() -> None:
    frame = make_synthetic_ohlcv(300, seed=1)
    config = FeatureConfig(ema_periods=[8, 21], volatility_windows=[12, 24], volume_z_window=24)
    original = build_features(frame, config).frame

    changed = frame.copy()
    changed.loc[250:, "close"] *= 10
    changed.loc[250:, "high"] *= 10
    changed.loc[250:, "low"] *= 10
    changed.loc[250:, "open"] *= 10
    mutated = build_features(changed, config).frame

    columns = build_features(frame, config).columns
    np.testing.assert_allclose(
        original.loc[:249, columns].to_numpy(),
        mutated.loc[:249, columns].to_numpy(),
        equal_nan=True,
    )


def test_k_ahead_dead_zone_labels() -> None:
    frame = make_synthetic_ohlcv(30, seed=2)
    frame["close"] = np.linspace(100.0, 130.0, len(frame))
    labels = build_labels(
        frame,
        LabelConfig(method="k_ahead", horizon_bars=2, dead_zone_bps=1.0),
    )
    assert (labels.labels.iloc[:-2] == 1).all()
    assert labels.labels.iloc[-2:].isna().all()


def test_triple_barrier_same_bar_double_hit_is_neutral() -> None:
    frame = make_synthetic_ohlcv(10, seed=3)
    frame.loc[1, "high"] = frame.loc[0, "close"] * 1.02
    frame.loc[1, "low"] = frame.loc[0, "close"] * 0.98
    labels = build_labels(
        frame,
        LabelConfig(
            method="triple_barrier",
            horizon_bars=2,
            take_profit_bps=100,
            stop_loss_bps=100,
        ),
    )
    assert labels.labels.iloc[0] == 0


def test_rsi_boundary_values_are_not_neutralized() -> None:
    rising = make_synthetic_ohlcv(80, seed=20)
    rising["close"] = np.linspace(100.0, 180.0, len(rising))
    rising["open"] = rising["close"]
    rising["high"] = rising["close"] * 1.001
    rising["low"] = rising["close"] * 0.999
    config = FeatureConfig(ema_periods=[8], rsi_period=14, volatility_windows=[12], volume_z_window=12)
    rising_features = build_features(rising, config).frame
    assert rising_features["rsi_14"].dropna().iloc[-1] == 100.0

    falling = rising.copy()
    falling["close"] = np.linspace(180.0, 100.0, len(falling))
    falling["open"] = falling["close"]
    falling["high"] = falling["close"] * 1.001
    falling["low"] = falling["close"] * 0.999
    falling_features = build_features(falling, config).frame
    assert falling_features["rsi_14"].dropna().iloc[-1] == 0.0
