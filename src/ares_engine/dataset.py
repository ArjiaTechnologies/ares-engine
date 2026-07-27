"""Sequence construction and chronological split utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class SequenceDataset:
    X: np.ndarray
    labels: np.ndarray
    timestamps: pd.DatetimeIndex
    closes: np.ndarray
    bar_returns: np.ndarray
    source_rows: np.ndarray
    feature_columns: list[str]

    @property
    def directional_mask(self) -> np.ndarray:
        mask: np.ndarray = np.isfinite(self.labels) & (self.labels != 0)
        return mask

    @property
    def y_binary(self) -> np.ndarray:
        output = np.full(len(self.labels), np.nan, dtype="float64")
        mask = self.directional_mask
        output[mask] = (self.labels[mask] > 0).astype("float64")
        return output


@dataclass(frozen=True, slots=True)
class Fold:
    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


class PurgedWalkForwardSplitter:
    """Expanding-window chronological splits with a purge before validation."""

    def __init__(
        self,
        *,
        min_train_size: int,
        validation_size: int,
        step_size: int,
        purge_size: int,
        max_splits: int,
    ) -> None:
        self.min_train_size = min_train_size
        self.validation_size = validation_size
        self.step_size = step_size
        self.purge_size = purge_size
        self.max_splits = max_splits

    def split(self, n_samples: int) -> Iterator[Fold]:
        validation_start = self.min_train_size + self.purge_size
        fold_number = 0
        while validation_start + self.validation_size <= n_samples and fold_number < self.max_splits:
            train_end = validation_start - self.purge_size
            train_indices = np.arange(0, train_end, dtype="int64")
            validation_indices = np.arange(
                validation_start, validation_start + self.validation_size, dtype="int64"
            )
            yield Fold(fold_number, train_indices, validation_indices)
            validation_start += self.step_size
            fold_number += 1


def build_sequence_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    labels: pd.Series,
    *,
    lookback_bars: int,
) -> SequenceDataset:
    """Create a sample at every complete bar; neutral labels remain for backtesting."""
    values = frame[feature_columns].to_numpy(dtype="float64")
    label_values = labels.to_numpy(dtype="float64")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    closes = frame["close"].to_numpy(dtype="float64")
    bar_returns_full = pd.Series(closes).pct_change().to_numpy(dtype="float64")

    sequences: list[np.ndarray] = []
    output_labels: list[float] = []
    output_timestamps: list[pd.Timestamp] = []
    output_closes: list[float] = []
    output_returns: list[float] = []
    source_rows: list[int] = []

    for end in range(lookback_bars - 1, len(frame)):
        start = end - lookback_bars + 1
        window = values[start : end + 1]
        if not np.isfinite(window).all():
            continue
        sequences.append(window)
        output_labels.append(label_values[end])
        output_timestamps.append(timestamps[end])
        output_closes.append(closes[end])
        output_returns.append(bar_returns_full[end])
        source_rows.append(end)

    if not sequences:
        raise ValueError("No complete sequences could be built; provide more history or shorter features")

    return SequenceDataset(
        X=np.asarray(sequences, dtype="float32"),
        labels=np.asarray(output_labels, dtype="float64"),
        timestamps=pd.DatetimeIndex(output_timestamps),
        closes=np.asarray(output_closes, dtype="float64"),
        bar_returns=np.asarray(output_returns, dtype="float64"),
        source_rows=np.asarray(source_rows, dtype="int64"),
        feature_columns=list(feature_columns),
    )


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    if X_train.ndim != 3:
        raise ValueError("Expected sequence tensor with shape (samples, lookback, features)")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def transform_sequences(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    original_shape = X.shape
    transformed = scaler.transform(X.reshape(-1, X.shape[-1]))
    reshaped: np.ndarray = transformed.reshape(original_shape).astype("float32")
    return reshaped
