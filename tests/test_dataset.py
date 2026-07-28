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
    dataset = build_sequence_dataset(
        features.frame, features.columns, labels.labels, lookback_bars=24
    )
    assert len(dataset.X) > 100
    assert (~dataset.directional_mask).any()
    assert dataset.X.shape[1:] == (24, len(features.columns))
    scaler = fit_scaler(dataset.X[:100])
    assert np.all(np.isfinite(scaler.mean_))


def test_minimum_required_bars_is_exact() -> None:
    from ares_engine.config import load_config
    from ares_engine.validation import feature_warmup_rows, minimum_required_bars, prepare_dataset

    config = load_config("configs/smoke.yaml")
    required = minimum_required_bars(config)
    warmup = feature_warmup_rows(config)
    # Empirical warm-up must match the arithmetic exactly.
    frame = make_synthetic_ohlcv(required, seed=90)
    features_built = build_features(frame, config.features)
    matrix = features_built.frame[features_built.columns]
    first_complete = int(matrix.notna().all(axis=1).idxmax())
    assert first_complete == warmup
    # At exactly the minimum, the split fits; one bar fewer and it cannot.
    _, dataset = prepare_dataset(frame, config)
    needed = (
        config.validation.min_train_bars
        + int(config.validation.purge_bars or 0)
        + config.validation.validation_bars
    )
    assert len(dataset.X) == needed
    _, smaller = prepare_dataset(frame.iloc[:-1], config)
    assert len(smaller.X) == needed - 1
