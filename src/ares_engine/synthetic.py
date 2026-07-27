"""Deterministic synthetic OHLCV for tests and offline demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import timeframe_to_seconds


def make_synthetic_ohlcv(
    bars: int = 5_000,
    *,
    timeframe: str = "1h",
    seed: int = 42,
    exchange: str = "synthetic",
    price_noise_bps: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.where((np.arange(bars) // 600) % 2 == 0, 0.00015, -0.00005)
    volatility = np.where((np.arange(bars) // 300) % 3 == 0, 0.008, 0.004)
    returns = regime + volatility * rng.normal(size=bars)
    close = 1_800.0 * np.exp(np.cumsum(returns))
    if price_noise_bps:
        close *= 1.0 + rng.normal(scale=price_noise_bps / 10_000.0, size=bars)
    open_price = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(loc=0.0025, scale=0.0012, size=bars))
    high = np.maximum(open_price, close) * (1.0 + spread)
    low = np.minimum(open_price, close) * (1.0 - spread)
    volume = rng.lognormal(mean=7.5, sigma=0.5, size=bars)
    frequency = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
    timestamp = pd.date_range("2024-01-01", periods=bars, freq=frequency, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "exchange": exchange,
            "symbol": "ETH/USD",
            "timeframe": timeframe,
        }
    )
