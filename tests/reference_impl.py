"""Independent reference implementations for the Fable 5 audit.

Deliberately written with explicit Python loops and floats (no pandas/numpy vectorization)
so that a shared library bug cannot make both implementations agree by accident.
"""

from __future__ import annotations

import math


# ---------------- backtest reference ----------------
def ref_backtest(
    bar_returns,
    probabilities,
    *,
    long_t,
    short_t,
    fee_bps,
    slip_bps,
    delay=1,
    cost_mult=1.0,
    periods_per_year=8766.0,
):
    n = len(bar_returns)
    assert len(probabilities) == n
    signals = []
    for p in probabilities:
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("invalid probability")
        if p >= long_t:
            signals.append(1.0)
        elif p <= short_t:
            signals.append(-1.0)
        else:
            signals.append(0.0)
    positions = [0.0] * n
    for i in range(n):
        j = i - delay
        positions[i] = signals[j] if j >= 0 else 0.0
    rate = (fee_bps + slip_bps) / 10_000.0 * cost_mult
    equity = 1.0
    prev_pos = 0.0
    equities, net_returns = [], []
    bankrupt = False
    bankruptcy_bar = None
    turnover = 0.0
    trades = 0
    active_bars = 0
    hits = 0
    for i in range(n):
        r = bar_returns[i]
        if not math.isfinite(r):
            raise ValueError("invalid bar return")
        change = abs(positions[i] - prev_pos)
        if change > 0:
            trades += 1
        turnover += change
        cost = change * rate
        gross = positions[i] * r
        raw_net = gross - cost
        if bankrupt:
            net = 0.0
            equity = 0.0
        elif raw_net <= -1.0 or equity * (1.0 + raw_net) <= 0.0:
            net = -1.0
            equity = 0.0
            bankrupt = True
            bankruptcy_bar = i
        else:
            net = raw_net
            equity *= 1.0 + net
        equities.append(equity)
        net_returns.append(net)
        if positions[i] != 0.0 and bankruptcy_bar is None:
            active_bars += 1
            if gross > 0:
                hits += 1
        prev_pos = positions[i]
    total_return = equity - 1.0
    peak = 1.0
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        dd = e / peak - 1.0
        max_dd = min(max_dd, dd)
    mean = sum(net_returns) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in net_returns) / n if n else 0.0
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(periods_per_year)) if (std > 0 and math.isfinite(std)) else 0.0
    if bankrupt:
        sharpe = min(sharpe, 0.0)
        active_bars = sum(position != 0.0 for position in positions[: (bankruptcy_bar or 0) + 1])
        turnover = sum(
            abs(positions[i] - (positions[i - 1] if i else 0.0))
            for i in range((bankruptcy_bar or 0) + 1)
        )
        trades = sum(
            abs(positions[i] - (positions[i - 1] if i else 0.0)) > 0
            for i in range((bankruptcy_bar or 0) + 1)
        )
    exposure = active_bars / n if n else 0.0
    hit_rate = 0.0 if bankrupt else hits / active_bars if active_bars else 0.0
    return {
        "positions": positions,
        "total_return": total_return,
        "final_equity": equity,
        "max_drawdown": abs(max_dd),
        "turnover": turnover,
        "trades": trades,
        "exposure": exposure,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "bankrupt": bankrupt,
        "bankruptcy_bar": bankruptcy_bar,
    }


# ---------------- feature references ----------------
def ref_ema(values, period):
    """EMA with span semantics, adjust=False, first period-1 outputs None."""
    alpha = 2.0 / (period + 1.0)
    out, ema = [], None
    for i, v in enumerate(values):
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        out.append(ema if i >= period - 1 else None)
    return out


def ref_wilder_rsi(values, period):
    """ewm(alpha=1/period, adjust=False) seeded at first delta, min_periods=period on deltas."""
    alpha = 1.0 / period
    out = [None]
    avg_gain = avg_loss = None
    count = 0
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if avg_gain is None:
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
        count += 1
        if count < period:
            out.append(None)
            continue
        if avg_loss == 0.0 and avg_gain > 0.0:
            out.append(100.0)
        elif avg_gain == 0.0 and avg_loss > 0.0:
            out.append(0.0)
        elif avg_gain == 0.0 and avg_loss == 0.0:
            out.append(50.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out


def ref_bollinger(values, period, n_std):
    mids, highs, lows, positions = [], [], [], []
    for i in range(len(values)):
        if i < period - 1:
            mids.append(None)
            highs.append(None)
            lows.append(None)
            positions.append(None)
            continue
        window = values[i - period + 1 : i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period  # ddof=0
        s = math.sqrt(var)
        hi, lo = m + n_std * s, m - n_std * s
        mids.append(m)
        highs.append(hi)
        lows.append(lo)
        width = hi - lo
        positions.append(((values[i] - lo) / width) if width != 0.0 else None)
    return mids, highs, lows, positions


def ref_log_returns(values):
    out = [None]
    for i in range(1, len(values)):
        out.append(math.log(values[i]) - math.log(values[i - 1]))
    return out


def ref_realized_vol(log_returns, window):
    out = []
    for i in range(len(log_returns)):
        chunk = log_returns[max(0, i - window + 1) : i + 1]
        if len(chunk) < window or any(c is None for c in chunk):
            out.append(None)
            continue
        m = sum(chunk) / window
        var = sum((x - m) ** 2 for x in chunk) / window
        out.append(math.sqrt(var))
    return out


def ref_volume_z(volumes, window):
    logv = [math.log1p(v) for v in volumes]
    out = []
    for i in range(len(logv)):
        chunk = logv[max(0, i - window + 1) : i + 1]
        if len(chunk) < window:
            out.append(None)
            continue
        m = sum(chunk) / window
        var = sum((x - m) ** 2 for x in chunk) / window
        s = math.sqrt(var)
        out.append(((logv[i] - m) / s) if s != 0.0 else None)
    return out


# ---------------- label references ----------------
def ref_k_ahead(closes, horizon, dead_zone_bps):
    dz = dead_zone_bps / 10_000.0
    out = []
    n = len(closes)
    for i in range(n):
        if i + horizon >= n:
            out.append(None)
            continue
        fr = closes[i + horizon] / closes[i] - 1.0
        out.append(1.0 if fr > dz else -1.0 if fr < -dz else 0.0)
    return out


def ref_triple_barrier(closes, highs, lows, horizon, tp_bps, sl_bps):
    tp = tp_bps / 10_000.0
    sl = sl_bps / 10_000.0
    n = len(closes)
    out = []
    for i in range(n):
        if i + horizon >= n:
            out.append(None)
            continue
        upper = closes[i] * (1 + tp)
        lower = closes[i] * (1 - sl)
        label = 0.0
        for j in range(i + 1, i + horizon + 1):
            hu = highs[j] >= upper
            hl = lows[j] <= lower
            if hu and hl:
                label = 0.0
                break
            if hu:
                label = 1.0
                break
            if hl:
                label = -1.0
                break
        out.append(label)
    return out
