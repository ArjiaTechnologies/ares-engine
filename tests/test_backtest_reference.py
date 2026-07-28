"""Backtest accounting verified against hand calculations and an independent reference."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from reference_impl import ref_backtest

from ares_engine.backtest import probabilities_to_signal, run_backtest
from ares_engine.config import BacktestConfig

HOURS_PER_YEAR = 365.25 * 24


def _ts(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")


def _cfg(**overrides) -> BacktestConfig:
    values = {
        "long_threshold": 0.6,
        "short_threshold": 0.4,
        "fee_bps": 10.0,
        "slippage_bps": 5.0,
        "execution_delay_bars": 1,
    }
    values.update(overrides)
    return BacktestConfig(**values)


def _run(returns, probabilities, config, cost_multiplier=1.0):
    return run_backtest(
        _ts(len(returns)),
        np.asarray(returns, dtype="float64"),
        np.asarray(probabilities, dtype="float64"),
        config,
        timeframe="1h",
        cost_multiplier=cost_multiplier,
    )


def _reference(returns, probabilities, config, cost_multiplier=1.0):
    return ref_backtest(
        list(returns),
        list(probabilities),
        long_t=config.long_threshold,
        short_t=config.short_threshold,
        fee_bps=config.fee_bps,
        slip_bps=config.slippage_bps,
        delay=config.execution_delay_bars,
        cost_mult=cost_multiplier,
        periods_per_year=HOURS_PER_YEAR,
    )


def _assert_matches_reference(returns, probabilities, config, cost_multiplier=1.0):
    result = _run(returns, probabilities, config, cost_multiplier)
    reference = _reference(returns, probabilities, config, cost_multiplier)
    ledger = result.ledger
    np.testing.assert_allclose(
        ledger["position"].to_numpy(), reference["positions"], atol=0, err_msg="positions"
    )
    for key in [
        "total_return",
        "final_equity",
        "max_drawdown",
        "turnover",
        "exposure",
        "hit_rate",
        "sharpe",
    ]:
        np.testing.assert_allclose(
            getattr(result.metrics, key), reference[key], rtol=1e-11, atol=1e-12, err_msg=key
        )
    assert result.metrics.trades == reference["trades"]
    assert result.metrics.bankrupt is reference["bankrupt"]
    assert result.metrics.bankruptcy_bar == reference["bankruptcy_bar"]
    return result


def test_always_flat_produces_no_trades_costs_or_drawdown() -> None:
    result = _assert_matches_reference([0.01, -0.02, 0.03, 0.0], [0.5] * 4, _cfg())
    assert result.metrics.trades == 0
    assert result.metrics.total_return == 0.0
    assert result.metrics.final_equity == 1.0
    assert result.metrics.exposure == 0.0
    assert result.metrics.max_drawdown == 0.0
    assert float(result.ledger["cost"].sum()) == 0.0


def test_enter_long_is_delayed_one_bar_and_charged_once() -> None:
    # Signal fires at bar 0; position exists on bar 1 only after the delay.
    config = _cfg(fee_bps=10.0, slippage_bps=0.0)
    result = _assert_matches_reference([0.0, 0.02, 0.0], [0.9, 0.5, 0.5], config)
    ledger = result.ledger
    assert list(ledger["position"]) == [0.0, 1.0, 0.0]
    # Hand calculation: cost bar1 = 1 * 0.001; equity = (1 + 0.02 - 0.001) * (1 - 0.001)
    expected_equity = (1.0 + 0.02 - 0.001) * (1.0 - 0.001)
    assert result.metrics.final_equity == pytest.approx(expected_equity, rel=1e-12)
    assert result.metrics.trades == 2  # entry and exit are separate position changes


def test_signal_on_final_bar_never_executes() -> None:
    result = _assert_matches_reference([0.0, 0.0, 0.5], [0.5, 0.5, 0.9], _cfg())
    assert list(result.ledger["position"]) == [0.0, 0.0, 0.0]
    assert result.metrics.trades == 0


def test_short_entry_and_exit_hand_calculation() -> None:
    config = _cfg(fee_bps=20.0, slippage_bps=0.0)
    result = _assert_matches_reference([0.0, -0.03, 0.0], [0.1, 0.5, 0.5], config)
    ledger = result.ledger
    assert list(ledger["position"]) == [0.0, -1.0, 0.0]
    expected_equity = (1.0 + 0.03 - 0.002) * (1.0 - 0.002)
    assert result.metrics.final_equity == pytest.approx(expected_equity, rel=1e-12)


def test_long_to_short_reversal_charges_double_turnover_once() -> None:
    config = _cfg(fee_bps=10.0, slippage_bps=5.0)
    result = _assert_matches_reference([0.0, 0.01, -0.02, 0.0], [0.9, 0.1, 0.5, 0.5], config)
    ledger = result.ledger
    assert list(ledger["position"]) == [0.0, 1.0, -1.0, 0.0]
    assert list(ledger["position_change"]) == [0.0, 1.0, 2.0, 1.0]
    rate = 15.0 / 10_000.0
    assert float(ledger.loc[2, "cost"]) == pytest.approx(2.0 * rate, rel=1e-12)
    assert result.metrics.turnover == pytest.approx(4.0)
    assert result.metrics.trades == 3


def test_short_to_long_reversal_mirror() -> None:
    result = _assert_matches_reference([0.0, -0.01, 0.02, 0.0], [0.1, 0.9, 0.5, 0.5], _cfg())
    assert list(result.ledger["position"]) == [0.0, -1.0, 1.0, 0.0]


def test_repeated_identical_signals_do_not_recharge_costs() -> None:
    config = _cfg(fee_bps=10.0, slippage_bps=0.0)
    result = _assert_matches_reference(
        [0.0, 0.01, 0.01, 0.01, 0.0], [0.9, 0.9, 0.9, 0.5, 0.5], config
    )
    ledger = result.ledger
    assert list(ledger["position"]) == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert float(ledger["cost"].gt(0).sum()) == 2  # entry and exit only


def test_threshold_equality_is_inclusive_on_both_sides() -> None:
    config = _cfg(long_threshold=0.6, short_threshold=0.4)
    signal = probabilities_to_signal(
        np.array([0.6, 0.4, 0.59999, 0.40001]), long_threshold=0.6, short_threshold=0.4
    )
    assert list(signal) == [1.0, -1.0, 0.0, 0.0]
    _assert_matches_reference([0.0, 0.01, -0.01, 0.0], [0.6, 0.4, 0.5, 0.5], config)


def test_nan_or_infinite_probability_fails_closed() -> None:
    for invalid in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            _run([0.0, 0.05], [invalid, 0.9], _cfg())


def test_nan_or_infinite_bar_return_fails_closed() -> None:
    for invalid in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            _run([0.0, invalid, 0.01], [0.9, 0.9, 0.5], _cfg())


def test_invalid_backtest_shapes_timestamps_probabilities_and_costs_fail_closed() -> None:
    config = _cfg()
    with pytest.raises(ValueError, match="identical lengths"):
        run_backtest(_ts(2), np.zeros(1), np.zeros(2), config, timeframe="1h")
    with pytest.raises(ValueError, match="at least one"):
        run_backtest(_ts(0), np.array([]), np.array([]), config, timeframe="1h")

    unsorted = pd.DatetimeIndex([_ts(2)[1], _ts(2)[0]])
    duplicated = pd.DatetimeIndex([_ts(1)[0], _ts(1)[0]])
    for timestamps in [unsorted, duplicated]:
        with pytest.raises(ValueError, match="sorted and unique"):
            run_backtest(timestamps, np.zeros(2), np.zeros(2), config, timeframe="1h")

    for probability in [-0.01, 1.01]:
        with pytest.raises(ValueError, match=r"within \[0, 1\]"):
            _run([0.0], [probability], config)
    for multiplier in [-0.01, np.nan, np.inf]:
        with pytest.raises(ValueError, match="finite and non-negative"):
            _run([0.0], [0.5], config, cost_multiplier=multiplier)


def test_fees_only_slippage_only_and_stress_multiplier() -> None:
    returns = [0.0, 0.02, -0.01, 0.0]
    probabilities = [0.9, 0.5, 0.1, 0.5]
    fees_only = _assert_matches_reference(returns, probabilities, _cfg(fee_bps=10, slippage_bps=0))
    slip_only = _assert_matches_reference(returns, probabilities, _cfg(fee_bps=0, slippage_bps=10))
    assert fees_only.metrics.final_equity == pytest.approx(
        slip_only.metrics.final_equity, rel=1e-12
    )
    base = _cfg(fee_bps=10, slippage_bps=5)
    single = _run(returns, probabilities, base, cost_multiplier=1.0)
    double = _run(returns, probabilities, base, cost_multiplier=2.0)
    _assert_matches_reference(returns, probabilities, base, cost_multiplier=2.0)
    assert double.metrics.final_equity < single.metrics.final_equity
    doubled_rate_equity = _run(
        returns, probabilities, _cfg(fee_bps=20, slippage_bps=10)
    ).metrics.final_equity
    assert double.metrics.final_equity == pytest.approx(doubled_rate_equity, rel=1e-12)


def test_gap_move_applies_full_bar_return_to_held_position() -> None:
    config = _cfg(fee_bps=0, slippage_bps=0)
    result = _assert_matches_reference([0.0, 0.30, 0.0], [0.9, 0.5, 0.5], config)
    assert result.metrics.total_return == pytest.approx(0.30, rel=1e-12)


def test_near_total_equity_loss_is_reported_not_hidden() -> None:
    config = _cfg(fee_bps=0, slippage_bps=0)
    result = _assert_matches_reference([0.0, -0.999, 0.0], [0.9, 0.5, 0.5], config)
    assert result.metrics.final_equity == pytest.approx(0.001, rel=1e-9)
    assert result.metrics.max_drawdown == pytest.approx(0.999, rel=1e-9)
    assert result.metrics.total_return == pytest.approx(-0.999, rel=1e-9)


def test_initial_loss_drawdown_is_measured_from_initial_capital() -> None:
    config = _cfg(fee_bps=0, slippage_bps=0, execution_delay_bars=1)
    result = _assert_matches_reference([-0.5, 0.0], [0.9, 0.5], config)
    # The delayed position is flat on bar zero, so use a cost-only first executed
    # bar to exercise the same initial-capital peak convention.
    result = _assert_matches_reference([0.0, -0.5], [0.9, 0.5], config)
    assert result.metrics.max_drawdown == pytest.approx(0.5)


def test_bankruptcy_metrics_cannot_benefit_from_later_winning_bars() -> None:
    config = _cfg(fee_bps=0, slippage_bps=0)
    base = _run([0.0, 1.1, 0.0], [0.1, 0.9, 0.9], config)
    extended = _run([0.0, 1.1, -0.9, -0.9, -0.9], [0.1, 0.9, 0.9, 0.9, 0.9], config)
    assert base.metrics.bankrupt and extended.metrics.bankrupt
    assert extended.metrics.final_equity == 0.0
    assert extended.metrics.max_drawdown == 1.0
    assert extended.metrics.hit_rate == 0.0
    assert extended.metrics.sharpe <= 0.0


@settings(max_examples=60, deadline=None)
@given(
    adverse_short_move=st.floats(min_value=1.000001, max_value=10.0, allow_nan=False),
    later_returns=st.lists(
        st.floats(min_value=-0.99, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=30,
    ),
)
def test_bankrupt_equity_is_terminal_for_random_future_paths(
    adverse_short_move: float, later_returns: list[float]
) -> None:
    returns = [0.0, adverse_short_move, *later_returns]
    probabilities = [0.1, *([0.9] * (len(returns) - 1))]
    result = _run(returns, probabilities, _cfg(fee_bps=0, slippage_bps=0))
    assert result.metrics.bankrupt
    assert result.metrics.bankruptcy_bar == 1
    assert result.ledger.loc[1:, "equity"].eq(0.0).all()
    assert result.ledger.loc[1:, "net_return"].iloc[1:].eq(0.0).all()
    assert result.metrics.total_return == -1.0
    assert result.metrics.max_drawdown == 1.0
    assert result.metrics.hit_rate == 0.0


def test_sharpe_annualization_matches_manual_formula() -> None:
    returns = [0.0, 0.01, -0.005, 0.02, 0.0]
    probabilities = [0.9, 0.9, 0.9, 0.9, 0.9]
    config = _cfg(fee_bps=0, slippage_bps=0)
    result = _run(returns, probabilities, config)
    net = result.ledger["net_return"].to_numpy()
    manual = net.mean() / net.std(ddof=0) * math.sqrt(HOURS_PER_YEAR)
    assert result.metrics.sharpe == pytest.approx(manual, rel=1e-12)


def test_zero_variance_returns_yield_zero_sharpe() -> None:
    result = _run([0.0] * 5, [0.5] * 5, _cfg())
    assert result.metrics.sharpe == 0.0


def test_execution_delay_two_bars_is_respected() -> None:
    config = _cfg(execution_delay_bars=2)
    result = _assert_matches_reference([0.0, 0.0, 0.03, 0.0], [0.9, 0.5, 0.5, 0.5], config)
    assert list(result.ledger["position"]) == [0.0, 0.0, 1.0, 0.0]


def test_randomized_agreement_with_reference_implementation() -> None:
    rng = np.random.default_rng(7)
    for _case in range(25):
        n = int(rng.integers(2, 120))
        returns = rng.normal(0, 0.02, size=n)
        probabilities = rng.uniform(0, 1, size=n)
        config = _cfg(
            long_threshold=float(rng.uniform(0.55, 0.8)),
            short_threshold=float(rng.uniform(0.2, 0.45)),
            fee_bps=float(rng.uniform(0, 25)),
            slippage_bps=float(rng.uniform(0, 25)),
            execution_delay_bars=int(rng.integers(1, 3)),
        )
        _assert_matches_reference(returns, probabilities, config)
