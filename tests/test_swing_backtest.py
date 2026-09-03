"""Triple-barrier engine for the swing setups.

The load-bearing tests are the ones about *not filling*: a setup whose entry order
never traded is not a trade, and an engine that counts it as one will report an edge
that does not exist. After that, the intrabar tie-break — because daily bars cannot
say whether the stop or the target came first, and the honest answer is a range.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import swing_backtest as sb
import swing_screener as ss


# ---------------------------------------------------------------------------
# Fill mechanics — hand-built bars, no randomness
# ---------------------------------------------------------------------------

def _bar(open_, high, low, close):
    return pd.Series({"Open": open_, "High": high, "Low": low, "Close": close,
                      "Volume": 1e6})


def _stop_buy(entry=100.0):
    return sb._Pending("AAA", "A/momentum", entry, 97.0, 103.0, 108.0, 10, 0, "")


def _limit_buy(entry=100.0):
    return sb._Pending("AAA", "B/ict", entry, 97.0, 103.0, 108.0, 10, 0, "")


def test_stop_buy_does_not_fill_below_its_trigger():
    assert sb._fill_price(_stop_buy(100.0), _bar(98, 99.5, 97, 99), 0.0) is None


def test_stop_buy_fills_when_price_trades_through():
    price = sb._fill_price(_stop_buy(100.0), _bar(98, 101, 97.5, 100.5), 0.0)
    assert price == pytest.approx(100.0)


def test_stop_buy_that_gaps_through_fills_at_the_open_and_worse():
    """A gap above the trigger does not fill at the trigger — it fills at the open."""
    price = sb._fill_price(_stop_buy(100.0), _bar(104, 106, 103.5, 105), 0.0)
    assert price == pytest.approx(104.0)
    assert price > 100.0


def test_limit_buy_does_not_fill_above_its_price():
    assert sb._fill_price(_limit_buy(100.0), _bar(103, 105, 101, 104), 0.0) is None


def test_limit_buy_fills_when_price_trades_down_into_it():
    price = sb._fill_price(_limit_buy(100.0), _bar(102, 103, 99.5, 101), 0.0)
    assert price == pytest.approx(100.0)


def test_limit_buy_that_gaps_below_fills_at_the_open_and_better():
    price = sb._fill_price(_limit_buy(100.0), _bar(96, 97.5, 95, 97), 0.0)
    assert price == pytest.approx(96.0)
    assert price < 100.0


def test_slippage_is_applied_against_the_buyer():
    clean = sb._fill_price(_stop_buy(100.0), _bar(98, 101, 97, 100), 0.0)
    slipped = sb._fill_price(_stop_buy(100.0), _bar(98, 101, 97, 100), 0.001)
    assert slipped > clean


def test_the_two_variants_rest_on_opposite_sides():
    assert _stop_buy().is_stop_buy is True
    assert _limit_buy().is_stop_buy is False


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def universe():
    bars = ss.load_demo(n_names=4, seed=11)
    bars.pop("SPY")
    return {s: f for s, f in bars.items()}


@pytest.fixture(scope="module")
def spy():
    bars = ss.load_demo(n_names=4, seed=11)
    return bars["SPY"]


@pytest.fixture(scope="module")
def config():
    return sb.SwingBacktestConfig(warmup=430)


@pytest.fixture(scope="module")
def result(universe, spy, config):
    return sb.run_swing_backtest(universe, spy, dict(ss.CFG), config)


def test_the_engine_reports_the_order_funnel(result):
    m = result.metrics
    assert m["orders_placed"] >= m["n_trades"], "more fills than orders is impossible"
    assert m["orders_placed"] == m["n_trades"] + m["orders_expired"] + m["orders_skipped_cash"]
    assert 0.0 <= m["fill_rate"] <= 1.0


def test_orders_that_never_triggered_are_not_trades(result):
    """The whole point of resting orders: a setup is not a position."""
    m = result.metrics
    if m["orders_placed"]:
        assert m["orders_expired"] > 0, "fixture never expired an order; test proves nothing"
    assert len(result.trades) == m["n_trades"]


def test_every_trade_has_a_complete_record(result):
    if result.trades.empty:
        pytest.skip("no trades in this run")
    assert list(result.trades.columns) == sb.TRADE_COLUMNS
    assert result.trades["entry_price"].gt(0).all()
    assert result.trades["exit_date"].ge(result.trades["entry_date"]).all()
    assert result.trades["bars_held"].ge(0).all()


def test_pnl_reconciles_with_the_equity_curve(result):
    total = float(result.trades["pnl"].sum())
    assert result.equity.iloc[-1] == pytest.approx(
        result.config.initial_equity + total, rel=1e-6)


def test_exit_reasons_are_all_accounted_for(result):
    if result.trades.empty:
        pytest.skip("no trades in this run")
    allowed = {"stop", "breakeven", "tp1", "tp2", "time_stop", "end_of_test"}
    assert set(result.trades["exit_reason"]) <= allowed


def test_adverse_excursion_is_negative_and_favourable_positive(result):
    if result.trades.empty:
        pytest.skip("no trades in this run")
    assert (result.trades["mae_r"] <= 1e-9).all()
    assert (result.trades["mfe_r"] >= -1e-9).all()


def test_a_stopped_trade_loses_about_one_r(result):
    """The stop defines 1R, so a clean stop-out should land near -1R."""
    stopped = result.trades[result.trades["exit_reason"] == "stop"]
    if stopped.empty:
        pytest.skip("no stop-outs in this run")
    assert stopped["r_multiple"].median() < -0.5
    assert stopped["r_multiple"].median() > -3.0


def test_a_partial_moves_the_stop_to_breakeven(result):
    """Trades that took the partial should not then lose much on the remainder."""
    partial = result.trades[result.trades["took_partial"]]
    if partial.empty:
        pytest.skip("no partials in this run")
    assert (partial["r_multiple"] > -0.6).all()


def test_the_time_stop_bounds_the_holding_period(result, config):
    if result.trades.empty:
        pytest.skip("no trades in this run")
    timed = result.trades[result.trades["exit_reason"] == "time_stop"]
    if not timed.empty:
        assert timed["bars_held"].max() <= config.max_hold + 2


def test_costs_drag_the_result_down(universe, spy, config):
    cheap = sb.run_swing_backtest(universe, spy, dict(ss.CFG),
                                  replace(config, slippage_bps=0.0, commission_bps=0.0))
    dear = sb.run_swing_backtest(universe, spy, dict(ss.CFG),
                                 replace(config, slippage_bps=60.0, commission_bps=10.0))
    if cheap.trades.empty:
        pytest.skip("no trades to compare")
    assert dear.metrics["total_costs"] > cheap.metrics["total_costs"]


def test_an_empty_universe_is_rejected(spy):
    with pytest.raises(ValueError):
        sb.run_swing_backtest({}, spy)


def test_a_universe_with_too_little_history_is_rejected(spy):
    tiny = {"AAA": ss.load_demo(n_names=1, seed=1)["SYN00"].head(50)}
    with pytest.raises(ValueError):
        sb.run_swing_backtest(tiny, spy, config=sb.SwingBacktestConfig(warmup=380))


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def test_truncating_the_data_reproduces_the_settled_trades(universe, spy, config):
    """Replaying on less history must not change trades that had already closed."""
    cut = 560
    truncated = {s: f.iloc[:cut] for s, f in universe.items()}
    cut_date = min(f.index[-1] for f in truncated.values())

    full = sb.run_swing_backtest(universe, spy, dict(ss.CFG), config)
    part = sb.run_swing_backtest(truncated, spy, dict(ss.CFG), config)

    def settled(res):
        df = res.trades
        df = df[(df["exit_date"] < cut_date) & (df["exit_reason"] != "end_of_test")]
        return df[["symbol", "entry_date", "exit_date", "exit_reason"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(settled(full), settled(part))


def test_capping_the_setup_window_changes_nothing(universe, spy, config):
    """The cap is a speed optimisation; it must not move a decision."""
    capped = sb.run_swing_backtest(universe, spy, dict(ss.CFG),
                                   replace(config, setup_window=420))
    uncapped = sb.run_swing_backtest(universe, spy, dict(ss.CFG),
                                     replace(config, setup_window=10_000))
    key = ["symbol", "entry_date", "exit_date", "exit_reason"]
    pd.testing.assert_frame_equal(capped.trades[key], uncapped.trades[key])


def test_precomputed_levels_match_computing_them_per_call(universe):
    """The level cache is only valid because pools use completed periods only."""
    cfg = dict(ss.CFG)
    mismatches = 0
    for frame in list(universe.values())[:3]:
        levels = ss.prior_period_levels(frame)
        for i in range(400, len(frame), 29):
            window = frame.iloc[max(0, i + 1 - 420):i + 1]
            a = ss.ict_setup(window, cfg)
            b = ss.ict_setup(window, cfg, levels)
            if (a is None) != (b is None):
                mismatches += 1
            elif a and b and a["entry"] != b["entry"]:
                mismatches += 1
    assert mismatches == 0


# ---------------------------------------------------------------------------
# The intrabar unknown
# ---------------------------------------------------------------------------

def test_the_tie_break_is_configurable_and_bounded(universe, spy, config):
    bound = sb.ambiguity_bound(universe, spy, dict(ss.CFG), config)
    assert bound["return_low"] <= bound["return_high"]
    assert bound["spread"] >= 0
    assert 0.0 <= bound["ambiguous_share"] <= 1.0
    assert bound["pessimistic"].config.intrabar == "stop"
    assert bound["optimistic"].config.intrabar == "target"


def test_optimistic_never_underperforms_pessimistic(universe, spy, config):
    """Resolving ties in your favour cannot produce a worse result."""
    bound = sb.ambiguity_bound(universe, spy, dict(ss.CFG), config)
    assert bound["optimistic"].metrics["total_return"] >= \
        bound["pessimistic"].metrics["total_return"] - 1e-9


def test_ambiguous_share_is_zero_when_no_trade_flips(universe, spy, config):
    bound = sb.ambiguity_bound(universe, spy, dict(ss.CFG), config)
    if bound["spread"] == 0:
        assert bound["ambiguous_share"] == 0.0


# ---------------------------------------------------------------------------
# The simulation layer applies unchanged
# ---------------------------------------------------------------------------

def test_the_simulation_layer_accepts_a_swing_result(result, universe):
    import simulation as sim
    if result.trades.empty:
        pytest.skip("no trades to simulate")
    report = sim.summarise(result, universe, sim.SimulationParams(n_paths=150, seed=3))
    assert report["bootstrap_summary"]["n_trades"] == len(result.trades)
    assert report["random_entry"]["n_trades"] == len(result.trades)
    assert "total_return" in report["buy_and_hold"]
