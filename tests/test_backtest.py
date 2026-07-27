import numpy as np
import pandas as pd

from ares_engine.backtest import run_backtest
from ares_engine.config import BacktestConfig


def test_execution_is_delayed_and_costed() -> None:
    timestamps = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    returns = np.array([0.0, 0.10, -0.10, 0.10, 0.0])
    probabilities = np.array([0.9, 0.9, 0.1, 0.1, 0.5])
    config = BacktestConfig(
        long_threshold=0.6,
        short_threshold=0.4,
        fee_bps=10,
        slippage_bps=0,
        execution_delay_bars=1,
    )
    result = run_backtest(timestamps, returns, probabilities, config, timeframe="1h")
    ledger = result.ledger
    assert ledger.loc[0, "position"] == 0
    assert ledger.loc[1, "position"] == 1
    assert ledger.loc[3, "position"] == -1
    assert ledger["cost"].sum() > 0
    assert result.metrics.trades >= 2
