"""Supervised labels based on future market movement, separate from indicators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LabelConfig


@dataclass(slots=True)
class LabelFrame:
    labels: pd.Series
    future_return: pd.Series


def k_ahead_labels(frame: pd.DataFrame, config: LabelConfig) -> LabelFrame:
    close = frame["close"].astype("float64")
    future_return = close.shift(-config.horizon_bars) / close - 1.0
    dead_zone = config.dead_zone_bps / 10_000.0
    labels = pd.Series(0.0, index=frame.index, name="label")
    labels = labels.mask(future_return > dead_zone, 1.0)
    labels = labels.mask(future_return < -dead_zone, -1.0)
    labels = labels.mask(future_return.isna(), np.nan)
    return LabelFrame(labels=labels, future_return=future_return.rename("future_return"))


def triple_barrier_labels(frame: pd.DataFrame, config: LabelConfig) -> LabelFrame:
    close = frame["close"].to_numpy(dtype="float64")
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    labels = np.full(len(frame), np.nan, dtype="float64")
    terminal_returns = np.full(len(frame), np.nan, dtype="float64")
    take_profit = config.take_profit_bps / 10_000.0
    stop_loss = config.stop_loss_bps / 10_000.0

    for index in range(len(frame) - config.horizon_bars):
        base = close[index]
        upper = base * (1.0 + take_profit)
        lower = base * (1.0 - stop_loss)
        label = 0.0
        for future_index in range(index + 1, index + config.horizon_bars + 1):
            hit_upper = high[future_index] >= upper
            hit_lower = low[future_index] <= lower
            if hit_upper and hit_lower:
                # Intrabar ordering is unknowable from OHLC alone. Neutral is honest; guessing is not.
                label = 0.0
                break
            if hit_upper:
                label = 1.0
                break
            if hit_lower:
                label = -1.0
                break
        labels[index] = label
        terminal_returns[index] = close[index + config.horizon_bars] / base - 1.0

    return LabelFrame(
        labels=pd.Series(labels, index=frame.index, name="label"),
        future_return=pd.Series(terminal_returns, index=frame.index, name="future_return"),
    )


def build_labels(frame: pd.DataFrame, config: LabelConfig) -> LabelFrame:
    if config.method == "k_ahead":
        return k_ahead_labels(frame, config)
    if config.method == "triple_barrier":
        return triple_barrier_labels(frame, config)
    raise ValueError(f"Unsupported label method: {config.method}")
