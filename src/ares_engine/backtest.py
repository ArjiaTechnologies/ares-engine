"""Cost-aware threshold backtest with delayed execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .utils import timeframe_to_seconds


@dataclass(slots=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    trades: int
    exposure: float
    hit_rate: float
    final_equity: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class BacktestResult:
    metrics: BacktestMetrics
    ledger: pd.DataFrame


def probabilities_to_signal(
    probabilities: np.ndarray,
    *,
    long_threshold: float,
    short_threshold: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype="float64").reshape(-1)
    signal = np.zeros_like(probabilities)
    signal[probabilities >= long_threshold] = 1.0
    signal[probabilities <= short_threshold] = -1.0
    return signal


def _safe_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    std = float(returns.std(ddof=0))
    if std <= 0 or not np.isfinite(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def run_backtest(
    timestamps: pd.DatetimeIndex,
    bar_returns: np.ndarray,
    probabilities: np.ndarray,
    config: BacktestConfig,
    *,
    timeframe: str,
    cost_multiplier: float = 1.0,
) -> BacktestResult:
    if len(timestamps) != len(bar_returns) or len(timestamps) != len(probabilities):
        raise ValueError("timestamps, returns, and probabilities must have identical lengths")

    ledger = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bar_return": np.nan_to_num(np.asarray(bar_returns, dtype="float64"), nan=0.0),
            "probability": np.asarray(probabilities, dtype="float64").reshape(-1),
        }
    )
    ledger["signal"] = probabilities_to_signal(
        ledger["probability"].to_numpy(),
        long_threshold=config.long_threshold,
        short_threshold=config.short_threshold,
    )
    ledger["position"] = ledger["signal"].shift(config.execution_delay_bars).fillna(0.0)
    ledger["position_change"] = ledger["position"].diff().abs().fillna(ledger["position"].abs())
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0 * cost_multiplier
    ledger["cost"] = ledger["position_change"] * cost_rate
    ledger["gross_return"] = ledger["position"] * ledger["bar_return"]
    ledger["net_return"] = ledger["gross_return"] - ledger["cost"]
    ledger["equity"] = (1.0 + ledger["net_return"]).cumprod()
    running_peak = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / running_peak - 1.0

    periods_per_year = 365.25 * 24 * 3600 / timeframe_to_seconds(timeframe)
    total_return = float(ledger["equity"].iloc[-1] - 1.0)
    years = max(len(ledger) / periods_per_year, 1.0 / periods_per_year)
    annualized_return = float(max(ledger["equity"].iloc[-1], 1e-12) ** (1.0 / years) - 1.0)
    active = ledger["position"] != 0
    active_hits = (ledger.loc[active, "gross_return"] > 0).mean() if active.any() else 0.0
    metrics = BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=_safe_sharpe(ledger["net_return"], periods_per_year),
        max_drawdown=float(abs(ledger["drawdown"].min())),
        turnover=float(ledger["position_change"].sum()),
        trades=int((ledger["position_change"] > 0).sum()),
        exposure=float(active.mean()),
        hit_rate=float(active_hits),
        final_equity=float(ledger["equity"].iloc[-1]),
    )
    return BacktestResult(metrics=metrics, ledger=ledger)
