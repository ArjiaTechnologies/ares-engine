"""Bounded, deterministic Keras lifecycle checks for both model families."""

import numpy as np
import pytest
from test_bundle_security import _training_config

from ares_engine.bundles import load_bundle
from ares_engine.config import load_config
from ares_engine.models import build_model, class_weights, fit_model, predict_probabilities
from ares_engine.synthetic import make_synthetic_ohlcv
from ares_engine.training import train_candidate

pytestmark = pytest.mark.ml


def test_cpu_only_execution() -> None:
    import tensorflow as tf

    assert tf.config.list_physical_devices("GPU") == []


def test_class_weights_balance_and_one_class_handling() -> None:
    balanced = class_weights(np.array([0, 1, 0, 1], dtype="float32"))
    assert balanced == {0: 1.0, 1: 1.0}
    skewed = class_weights(np.array([0, 0, 0, 1], dtype="float32"))
    assert skewed is not None
    assert skewed[1] == pytest.approx(2.0)
    assert skewed[0] == pytest.approx(2.0 / 3.0)
    assert class_weights(np.array([1, 1, 1], dtype="float32")) is None


@pytest.mark.parametrize("family", ["lstm", "tcn"])
def test_same_seed_builds_identical_models(family) -> None:
    config = load_config("configs/smoke.yaml").model
    config.family = family
    config.hidden_units = 8
    config.tcn_blocks = 2
    probe = np.random.default_rng(3).normal(size=(4, 12, 5)).astype("float32")
    first = predict_probabilities(build_model((12, 5), config, seed=11), probe)
    second = predict_probabilities(build_model((12, 5), config, seed=11), probe)
    third = predict_probabilities(build_model((12, 5), config, seed=12), probe)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third), "different seeds must not produce identical weights"
    assert first.shape == (4,)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_fit_model_wires_early_stopping_nan_guard_and_no_shuffle() -> None:
    captured: dict = {}

    class RecordingModel:
        def fit(self, X, y, **kwargs):
            captured.update(kwargs)
            captured["shapes"] = (X.shape, y.shape)
            return "history"

    config = load_config("configs/smoke.yaml").model
    X = np.zeros((10, 4, 3), dtype="float32")
    y = np.array([0, 1] * 5, dtype="float32")
    Xv = np.zeros((4, 4, 3), dtype="float32")
    yv = np.array([0, 1, 0, 1], dtype="float32")
    fit_model(RecordingModel(), X, y, config, X_validation=Xv, y_validation=yv)
    callbacks = captured["callbacks"]
    early = next(c for c in callbacks if type(c).__name__ == "EarlyStopping")
    assert early.monitor == "val_auc"
    assert getattr(early, "restore_best_weights", None) is True
    assert any(type(c).__name__ == "TerminateOnNaN" for c in callbacks)
    assert captured["shuffle"] is False
    assert captured["batch_size"] == config.batch_size
    assert captured["validation_data"] == (Xv, yv)
    assert captured["class_weight"] == {0: 1.0, 1: 1.0}

    captured.clear()
    fit_model(RecordingModel(), X, y, config)
    early = next(c for c in captured["callbacks"] if type(c).__name__ == "EarlyStopping")
    assert early.monitor == "loss"
    assert captured["validation_data"] is None


def test_tcn_lifecycle_trains_exports_and_reloads_with_identical_predictions(
    tmp_path_factory,
) -> None:
    root = tmp_path_factory.mktemp("tcn-lifecycle")
    config = _training_config(root)
    config.model.family = "tcn"
    config.model.tcn_blocks = 2
    config.model.tcn_kernel_size = 3
    frame = make_synthetic_ohlcv(430, seed=61, exchange="coinbase")
    bundle, metrics = train_candidate(frame, config, bundle_name="tcn_bundle")
    assert metrics["passed"] is True
    loaded = load_bundle(bundle)
    assert loaded.config.model.family == "tcn"
    probe = (
        np.random.default_rng(5).normal(size=(3, config.model.lookback_bars, 16)).astype("float32")
    )
    first = predict_probabilities(loaded.model, probe)
    second = predict_probabilities(load_bundle(bundle).model, probe)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    assert np.all(np.isfinite(first))


def test_tcn_receptive_field_and_causal_padding_structure() -> None:
    config = load_config("configs/smoke.yaml").model
    config.family = "tcn"
    config.tcn_blocks = 3
    config.tcn_kernel_size = 3
    config.hidden_units = 8
    model = build_model((36, 5), config, seed=9)
    conv_layers = [layer for layer in model.layers if type(layer).__name__ == "Conv1D"]
    dilations = sorted(
        {tuple(layer.dilation_rate)[0] for layer in conv_layers if layer.kernel_size[0] > 1}
    )
    assert dilations == [1, 2, 4], "dilations must double per block"
    assert all(layer.padding == "causal" for layer in conv_layers if layer.kernel_size[0] > 1), (
        "every temporal convolution must be causal"
    )
