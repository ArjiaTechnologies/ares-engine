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
    bankrupt: bool
    bankruptcy_bar: int | None

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
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be within [0, 1]")
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

    if len(timestamps) == 0:
        raise ValueError("backtest requires at least one bar")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("timestamps must be sorted and unique")
    returns = np.asarray(bar_returns, dtype="float64").reshape(-1)
    if not np.isfinite(returns).all():
        raise ValueError("bar_returns must contain only finite values")
    if not np.isfinite(cost_multiplier) or cost_multiplier < 0:
        raise ValueError("cost_multiplier must be finite and non-negative")

    ledger = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bar_return": returns,
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
    ledger["raw_net_return"] = ledger["gross_return"] - ledger["cost"]
    effective_returns: list[float] = []
    equities: list[float] = []
    bankrupt_flags: list[bool] = []
    equity = 1.0
    bankrupt = False
    bankruptcy_bar: int | None = None
    for index, raw_return in enumerate(ledger["raw_net_return"].to_numpy(dtype="float64")):
        if bankrupt:
            effective_returns.append(0.0)
            equities.append(0.0)
            bankrupt_flags.append(True)
            continue
        if raw_return <= -1.0 or equity * (1.0 + raw_return) <= 0.0:
            bankrupt = True
            bankruptcy_bar = index
            effective_returns.append(-1.0)
            equity = 0.0
        else:
            effective_returns.append(float(raw_return))
            equity *= 1.0 + float(raw_return)
        equities.append(equity)
        bankrupt_flags.append(bankrupt)
    ledger["net_return"] = effective_returns
    ledger["equity"] = equities
    ledger["bankrupt"] = bankrupt_flags
    # Initial capital is part of the peak. Omitting it hides drawdown when the
    # very first executable bar loses money.
    running_peak = ledger["equity"].cummax().clip(lower=1.0)
    ledger["drawdown"] = ledger["equity"] / running_peak - 1.0

    periods_per_year = 365.25 * 24 * 3600 / timeframe_to_seconds(timeframe)
    total_return = float(ledger["equity"].iloc[-1] - 1.0)
    years = max(len(ledger) / periods_per_year, 1.0 / periods_per_year)
    if bankrupt:
        annualized_return = -1.0
    else:
        annualized_log_return = np.log(float(ledger["equity"].iloc[-1])) / years
        maximum_log = np.log(np.finfo("float64").max)
        annualized_return = float(np.expm1(min(annualized_log_return, maximum_log)))
    evaluated = ledger.iloc[: bankruptcy_bar + 1] if bankruptcy_bar is not None else ledger
    active = evaluated["position"] != 0
    active_hits = (evaluated.loc[active, "gross_return"] > 0).mean() if active.any() else 0.0
    sharpe = _safe_sharpe(ledger["net_return"], periods_per_year)
    if bankrupt:
        # Solvency is a hard gate, and no secondary metric may make an insolvent
        # candidate look attractive.
        sharpe = min(sharpe, 0.0)
        active_hits = 0.0
    metrics = BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        max_drawdown=float(abs(ledger["drawdown"].min())),
        turnover=float(evaluated["position_change"].sum()),
        trades=int((evaluated["position_change"] > 0).sum()),
        exposure=float(active.sum() / len(ledger)),
        hit_rate=float(active_hits),
        final_equity=float(ledger["equity"].iloc[-1]),
        bankrupt=bankrupt,
        bankruptcy_bar=bankruptcy_bar,
    )
    return BacktestResult(metrics=metrics, ledger=ledger)
