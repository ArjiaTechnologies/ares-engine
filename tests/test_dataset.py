import numpy as np

from ares_engine.config import FeatureConfig, LabelConfig
from ares_engine.dataset import PurgedWalkForwardSplitter, build_sequence_dataset, fit_scaler
from ares_engine.features import build_features
from ares_engine.labels import build_labels
from ares_engine.synthetic import make_synthetic_ohlcv


def test_purged_walk_forward_boundaries() -> None:
    folds = list(
        PurgedWalkForwardSplitter(
            min_train_size=100,
            validation_size=20,
            step_size=20,
            purge_size=5,
            max_splits=3,
        ).split(200)
    )
    assert len(folds) == 3
    assert folds[0].train_indices[-1] == 99
    assert folds[0].validation_indices[0] == 105
    assert set(folds[0].train_indices).isdisjoint(folds[0].validation_indices)


def test_sequence_dataset_keeps_neutral_bars_for_backtest() -> None:
    frame = make_synthetic_ohlcv(300, seed=4)
    features = build_features(
        frame,
        FeatureConfig(ema_periods=[8, 21], volatility_windows=[12, 24], volume_z_window=24),
    )
    labels = build_labels(
        features.frame,
        LabelConfig(method="k_ahead", horizon_bars=6, dead_zone_bps=10_000),
    )
    dataset = build_sequence_dataset(features.frame, features.columns, labels.labels, lookback_bars=24)
    assert len(dataset.X) > 100
    assert (~dataset.directional_mask).any()
    assert dataset.X.shape[1:] == (24, len(features.columns))
    scaler = fit_scaler(dataset.X[:100])
    assert np.all(np.isfinite(scaler.mean_))
