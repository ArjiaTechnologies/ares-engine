"""Keras LSTM and compact TCN sequence models, with TensorFlow as the primary backend."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from .config import ModelConfig
from .exceptions import AresError
from .utils import set_global_seed


def keras_api() -> Any:
    """Return tf.keras when available, otherwise Keras 3 on an explicit backend.

    TensorFlow remains the reference backend. The fallback exists so the same `.keras` architecture can
    be smoke-tested when a TensorFlow runtime is intentionally not installed.
    """
    try:
        import tensorflow as tf  # type: ignore

        return tf.keras
    except ImportError:
        if "keras" not in sys.modules:
            os.environ.setdefault("KERAS_BACKEND", os.getenv("ARES_KERAS_BACKEND", "torch"))
        try:
            import keras  # type: ignore
        except ImportError as exc:
            raise AresError(
                "No Keras runtime is installed. Install `.[ml]` for TensorFlow or `.[ml-torch]` "
                "for the Keras/Torch compatibility backend."
            ) from exc
        return keras


def backend_name() -> str:
    keras = keras_api()
    try:
        return str(keras.backend.backend())
    except Exception:
        return "tensorflow"


def build_model(input_shape: tuple[int, int], config: ModelConfig, *, seed: int) -> Any:
    keras = keras_api()
    set_global_seed(seed)
    layers = keras.layers
    inputs = layers.Input(shape=input_shape, name="sequence")

    if config.family == "lstm":
        x = layers.LSTM(config.hidden_units, name="lstm")(inputs)
        x = layers.Dropout(config.dropout, name="dropout")(x)
    elif config.family == "tcn":
        x = inputs
        for block in range(config.tcn_blocks):
            dilation = 2**block
            residual = x
            x = layers.Conv1D(
                filters=config.hidden_units,
                kernel_size=config.tcn_kernel_size,
                padding="causal",
                dilation_rate=dilation,
                name=f"tcn_conv_{block}_a",
            )(x)
            x = layers.LayerNormalization(name=f"tcn_norm_{block}_a")(x)
            x = layers.Activation("relu", name=f"tcn_relu_{block}_a")(x)
            x = layers.Dropout(config.dropout, name=f"tcn_dropout_{block}")(x)
            x = layers.Conv1D(
                filters=config.hidden_units,
                kernel_size=config.tcn_kernel_size,
                padding="causal",
                dilation_rate=dilation,
                name=f"tcn_conv_{block}_b",
            )(x)
            if residual.shape[-1] != config.hidden_units:
                residual = layers.Conv1D(config.hidden_units, 1, name=f"tcn_residual_{block}")(
                    residual
                )
            x = layers.Add(name=f"tcn_add_{block}")([x, residual])
            x = layers.Activation("relu", name=f"tcn_relu_{block}_b")(x)
        x = layers.GlobalAveragePooling1D(name="tcn_pool")(x)
    else:
        raise ValueError(f"Unsupported model family: {config.family}")

    outputs = layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"ares_{config.family}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def class_weights(y: np.ndarray) -> dict[int, float] | None:
    y = np.asarray(y, dtype="int64")
    counts = np.bincount(y, minlength=2)
    if counts.min() == 0:
        return None
    total = counts.sum()
    return {0: float(total / (2 * counts[0])), 1: float(total / (2 * counts[1]))}


def fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ModelConfig,
    *,
    X_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
    verbose: int = 0,
) -> Any:
    keras = keras_api()
    validation_data = None
    has_validation = (
        X_validation is not None
        and y_validation is not None
        and len(X_validation) > 0
        and len(y_validation) > 0
    )
    if has_validation:
        validation_data = (X_validation, y_validation)
    callbacks: list[Any] = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc" if has_validation else "loss",
            mode="max" if has_validation else "min",
            patience=config.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
    return model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=False,
        class_weight=class_weights(y_train),
        callbacks=callbacks,
        verbose=verbose,
    )


def predict_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    predictions = model.predict(X, verbose=0)
    probabilities = np.asarray(predictions, dtype="float64").reshape(-1)
    if len(probabilities) != len(X):
        raise AresError("Model returned a prediction count that does not match its input")
    if not np.isfinite(probabilities).all():
        raise AresError("Model returned NaN or infinite probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise AresError("Model returned probabilities outside [0, 1]")
    return probabilities


def clear_session() -> None:
    keras_api().backend.clear_session()
