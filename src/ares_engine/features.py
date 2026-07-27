"""Leak-free feature construction from cleaned OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeatureConfig


@dataclass(slots=True)
class FeatureFrame:
    frame: pd.DataFrame
    columns: list[str]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    # Wilder RSI has explicit boundary values. Treating a zero-loss uptrend as neutral (50) is wrong.
    rsi = rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    rsi = rsi.mask((average_gain == 0.0) & (average_loss > 0.0), 0.0)
    rsi = rsi.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)
    return rsi.where(average_gain.notna() & average_loss.notna())


def build_features(ohlcv: pd.DataFrame, config: FeatureConfig) -> FeatureFrame:
    """Build only backward-looking features; no centered windows or future fills."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"Cannot build features; missing columns: {sorted(missing)}")

    frame = ohlcv.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    close = frame["close"].astype("float64")
    volume = frame["volume"].astype("float64")
    feature_columns: list[str] = []

    frame["close"] = close
    feature_columns.append("close")

    for period in config.ema_periods:
        ema_name = f"ema_{period}"
        distance_name = f"close_to_ema_{period}"
        frame[ema_name] = close.ewm(span=period, adjust=False, min_periods=period).mean()
        frame[distance_name] = close / frame[ema_name] - 1.0
        feature_columns.extend([ema_name, distance_name])

    rsi_name = f"rsi_{config.rsi_period}"
    frame[rsi_name] = _rsi(close, config.rsi_period)
    feature_columns.append(rsi_name)

    mid_name = f"bb_mid_{config.bollinger_period}"
    high_name = f"bb_high_{config.bollinger_period}"
    low_name = f"bb_low_{config.bollinger_period}"
    position_name = f"bb_position_{config.bollinger_period}"
    rolling = close.rolling(config.bollinger_period, min_periods=config.bollinger_period)
    frame[mid_name] = rolling.mean()
    std = rolling.std(ddof=0)
    frame[high_name] = frame[mid_name] + config.bollinger_std * std
    frame[low_name] = frame[mid_name] - config.bollinger_std * std
    width = (frame[high_name] - frame[low_name]).replace(0.0, np.nan)
    frame[position_name] = (close - frame[low_name]) / width
    feature_columns.extend([mid_name, high_name, low_name, position_name])

    frame["log_return_1"] = np.log(close).diff()
    feature_columns.append("log_return_1")

    for window in config.volatility_windows:
        name = f"realized_vol_{window}"
        frame[name] = frame["log_return_1"].rolling(window, min_periods=window).std(ddof=0)
        feature_columns.append(name)

    frame["log_volume"] = np.log1p(volume)
    volume_mean = frame["log_volume"].rolling(
        config.volume_z_window, min_periods=config.volume_z_window
    ).mean()
    volume_std = frame["log_volume"].rolling(
        config.volume_z_window, min_periods=config.volume_z_window
    ).std(ddof=0)
    volume_z_name = f"volume_z_{config.volume_z_window}"
    frame[volume_z_name] = (frame["log_volume"] - volume_mean) / volume_std.replace(0.0, np.nan)
    feature_columns.extend(["log_volume", volume_z_name])

    frame["high_low_range"] = (frame["high"] - frame["low"]) / close
    frame["close_open_return"] = frame["close"] / frame["open"] - 1.0
    feature_columns.extend(["high_low_range", "close_open_return"])

    return FeatureFrame(frame=frame, columns=feature_columns)
