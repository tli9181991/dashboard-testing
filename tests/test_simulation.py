"""Monte Carlo layer: sequence risk, and whether the signal beats random exposure."""

import numpy as np
import pandas as pd
import pytest

import data as data_mod
from backtest import BacktestConfig, run_backtest
from simulation import (
    SimulationParams,
    bootstrap_paths,
    buy_and_hold,
    max_drawdowns,
    random_entry_benchmark,
    summarise,
)


@pytest.fixture(scope="module")
def universe():
    return {
        "AAA": data_mod.synthetic_ohlcv(n=600, seed=42),
        "BBB": data_mod.synthetic_ohlcv(n=600, seed=99, annual_drift=0.0, annual_vol=0.55),
    }


@pytest.fixture(scope="module")
def result(universe):
    index = universe["AAA"].index
    bench = data_mod.synthetic_benchmark(index, seed=5, regime_flip=380)
    return run_backtest(universe, bench, BacktestConfig(warmup=210))


@pytest.fixture(scope="module")
def fast():
    return SimulationParams(n_paths=200, seed=3)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_paths_start_at_one_and_have_a_point_per_trade(fast):
    returns = [0.05, -0.02, 0.03, -0.01, 0.04]
    res = bootstrap_paths(returns, fast)
    assert res.paths.shape == (fast.n_paths, len(returns) + 1)
    assert np.allclose(res.paths[:, 0], 1.0)


def test_a_uniformly_positive_strategy_never_loses(fast):
    res = bootstrap_paths([0.01] * 8, fast)
    assert (res.finals > 1).all()
    assert res.summary()["prob_profit"] == 1.0


def test_a_uniformly_negative_strategy_never_wins(fast):
    res = bootstrap_paths([-0.01] * 8, fast)
    assert (res.finals < 1).all()
    assert res.summary()["prob_profit"] == 0.0


def test_the_bootstrap_brackets_the_realised_outcome(fast):
    """The actual sequence is one draw from the distribution, so it must sit in it."""
    returns = [0.08, -0.03, 0.05, -0.04, 0.02, 0.06, -0.01]
    realised = float(np.prod([1 + r for r in returns]))
    res = bootstrap_paths(returns, fast)
    assert res.finals.min() <= realised <= res.finals.max()


def test_order_does_not_change_the_iid_distribution(fast):
    """Compounding is order-independent, so a shuffle must resample the same set."""
    returns = [0.05, -0.02, 0.03, -0.01]
    a = bootstrap_paths(returns, fast).finals
    b = bootstrap_paths(list(reversed(returns)), fast).finals
    assert np.isclose(np.median(a), np.median(b), rtol=0.15)


def test_too_few_trades_returns_nothing(fast):
    assert bootstrap_paths([], fast) is None
    assert bootstrap_paths([0.05], fast) is None


def test_the_seed_makes_it_reproducible():
    p = SimulationParams(n_paths=100, seed=11)
    a = bootstrap_paths([0.03, -0.01, 0.02], p).finals
    b = bootstrap_paths([0.03, -0.01, 0.02], p).finals
    np.testing.assert_array_equal(a, b)


def test_block_resampling_keeps_the_requested_length(fast):
    blocked = SimulationParams(n_paths=fast.n_paths, seed=fast.seed, block_size=3)
    res = bootstrap_paths([0.05, -0.02, 0.03, -0.01, 0.04, 0.01], blocked)
    assert res.paths.shape[1] == 7


def test_summary_reports_a_band_in_order(fast):
    res = bootstrap_paths([0.06, -0.03, 0.04, -0.02, 0.05], fast)
    s = res.summary()
    assert s["p05_return"] <= s["median_return"] <= s["p95_return"]
    assert 0.0 <= s["prob_profit"] <= 1.0
    assert s["median_max_drawdown"] <= 0.0
    assert s["worst_max_drawdown"] <= s["median_max_drawdown"]


def test_drawdown_of_a_monotonic_rise_is_zero():
    rising = np.array([[1.0, 1.1, 1.2, 1.3]])
    assert max_drawdowns(rising)[0] == pytest.approx(0.0)


def test_drawdown_measures_peak_to_trough():
    path = np.array([[1.0, 2.0, 1.0, 1.5]])
    assert max_drawdowns(path)[0] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Buy and hold
# ---------------------------------------------------------------------------

def test_buy_and_hold_matches_a_hand_computed_return():
    idx = pd.bdate_range("2024-01-01", periods=3)
    frame = pd.DataFrame({"Open": [10, 11, 12], "High": [10, 11, 12], "Low": [10, 11, 12],
                          "Close": [10.0, 11.0, 12.0], "Volume": [1e6] * 3}, index=idx)
    bh = buy_and_hold({"AAA": frame})
    assert bh["total_return"] == pytest.approx(0.2)
    assert bh["per_symbol"]["AAA"] == pytest.approx(0.2)


def test_buy_and_hold_equal_weights_the_names():
    idx = pd.bdate_range("2024-01-01", periods=2)
    up = pd.DataFrame({"Open": [10, 12], "High": [10, 12], "Low": [10, 12],
                       "Close": [10.0, 12.0], "Volume": [1e6] * 2}, index=idx)
    flat = pd.DataFrame({"Open": [10, 10], "High": [10, 10], "Low": [10, 10],
                         "Close": [10.0, 10.0], "Volume": [1e6] * 2}, index=idx)
    bh = buy_and_hold({"UP": up, "FLAT": flat})
    assert bh["total_return"] == pytest.approx(0.10)


def test_buy_and_hold_on_nothing(universe):
    assert buy_and_hold({}) == {}


def test_buy_and_hold_reports_a_drawdown(universe):
    bh = buy_and_hold(universe)
    assert bh["max_drawdown"] <= 0.0
    assert len(bh["curve"]) > 0


# ---------------------------------------------------------------------------
# Random entry — the signal test
# ---------------------------------------------------------------------------

def test_random_entry_places_the_strategy_in_a_distribution(universe, result, fast):
    bench = random_entry_benchmark(universe, result.trades, fast)
    assert bench is not None
    assert 0.0 <= bench["percentile"] <= 100.0
    assert bench["distribution"].size == fast.n_paths
    assert bench["n_trades"] == len(result.trades)


def test_random_entry_uses_the_strategy_holding_periods(universe, result, fast):
    """Same exposure profile, different entry dates — that is the whole comparison."""
    bench = random_entry_benchmark(universe, result.trades, fast)
    assert bench["n_trades"] == len(result.trades["bars_held"])


def test_a_strategy_far_above_the_random_draw_scores_high(universe, fast):
    """A synthetic result whose trades are implausibly good must land at the top."""
    trades = pd.DataFrame({
        "return_pct": [0.60] * 6,
        "bars_held": [5] * 6,
    })
    bench = random_entry_benchmark(universe, trades, fast)
    assert bench["percentile"] > 95


def test_a_strategy_far_below_the_random_draw_scores_low(universe, fast):
    trades = pd.DataFrame({
        "return_pct": [-0.60] * 6,
        "bars_held": [5] * 6,
    })
    bench = random_entry_benchmark(universe, trades, fast)
    assert bench["percentile"] < 5


def test_random_entry_needs_trades(universe, fast):
    empty = pd.DataFrame(columns=["return_pct", "bars_held"])
    assert random_entry_benchmark(universe, empty, fast) is None
    assert random_entry_benchmark({}, pd.DataFrame({"return_pct": [0.1, 0.2],
                                                    "bars_held": [3, 4]}), fast) is None


def test_costs_drag_the_random_benchmark_down(universe, result, fast):
    cheap = random_entry_benchmark(universe, result.trades, fast, cost_bps=0.0)
    dear = random_entry_benchmark(universe, result.trades, fast, cost_bps=100.0)
    assert dear["random_mean_trade"] < cheap["random_mean_trade"]


# ---------------------------------------------------------------------------
# The combined report
# ---------------------------------------------------------------------------

def test_summarise_runs_all_three_comparisons(universe, result, fast):
    report = summarise(result, universe, fast)
    assert set(report) >= {"metrics", "bootstrap", "bootstrap_summary",
                           "buy_and_hold", "random_entry"}
    assert report["metrics"]["n_trades"] == len(result.trades)
    assert report["bootstrap_summary"]["n_trades"] == len(result.trades)
    assert report["buy_and_hold"]["total_return"] is not None
