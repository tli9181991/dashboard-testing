"""Chart builders for the backtest and its simulation layer."""

import numpy as np
import pandas as pd
import pytest

import backtest_charts as charts
import data as data_mod
import simulation as sim
import viz
from backtest import BacktestConfig, run_backtest


@pytest.fixture(scope="module")
def scenario():
    universe = {
        "AAA": data_mod.synthetic_ohlcv(n=700, seed=42),
        "BBB": data_mod.synthetic_ohlcv(n=700, seed=99, annual_drift=0.0, annual_vol=0.55),
    }
    bench = data_mod.synthetic_benchmark(universe["AAA"].index, seed=5, regime_flip=430)
    result = run_backtest(universe, bench, BacktestConfig(warmup=210))
    if result.trades.empty:
        pytest.skip("synthetic run produced no trades")
    report = sim.summarise(result, universe, sim.SimulationParams(n_paths=200, seed=3))
    return result, universe, report


@pytest.fixture
def every_figure(scenario):
    result, _, report = scenario
    return [
        charts.build_equity_comparison(result, report["buy_and_hold"]),
        charts.build_bootstrap_fan(report["bootstrap"], result.trades["return_pct"]),
        charts.build_random_entry_distribution(report["random_entry"]),
        charts.build_per_symbol_returns(result, report["buy_and_hold"]),
    ]


# ---------------------------------------------------------------------------
# Chrome & palette discipline
# ---------------------------------------------------------------------------

def test_every_chart_paints_the_shared_surface(every_figure):
    for fig in every_figure:
        assert fig.layout.paper_bgcolor == viz.SURFACE
        assert fig.layout.plot_bgcolor == viz.SURFACE


def test_gridlines_are_solid_hairlines(every_figure):
    for fig in every_figure:
        assert fig.layout.xaxis.griddash == "solid"
        assert fig.layout.yaxis.gridcolor == viz.GRID


def test_no_chart_uses_two_y_axes(every_figure):
    for fig in every_figure:
        assert "yaxis2" not in fig.layout, "dual axes invent correlations"


def test_charts_with_two_series_name_both(every_figure):
    for fig in every_figure:
        named = [t.name for t in fig.data if t.name]
        if len(named) >= 2:
            assert len(set(named)) == len(named), "duplicate legend entries"


# ---------------------------------------------------------------------------
# Equity comparison
# ---------------------------------------------------------------------------

def test_equity_comparison_plots_both_lines(scenario):
    result, _, report = scenario
    fig = charts.build_equity_comparison(result, report["buy_and_hold"])
    assert [t.name for t in fig.data] == ["Strategy", "Buy & hold"]
    assert fig.data[0].line.color == viz.CATEGORICAL[0]
    assert fig.data[1].line.color == viz.CATEGORICAL[1]


def test_equity_comparison_indexes_the_strategy_to_one(scenario):
    result, _, report = scenario
    fig = charts.build_equity_comparison(result, report["buy_and_hold"])
    assert float(fig.data[0].y[0]) == pytest.approx(1.0)


def test_equity_comparison_survives_a_missing_benchmark(scenario):
    result, _, _ = scenario
    fig = charts.build_equity_comparison(result, {})
    assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# Bootstrap fan
# ---------------------------------------------------------------------------

def test_fan_draws_two_bands_a_median_and_the_realised_path(scenario):
    result, _, report = scenario
    fig = charts.build_bootstrap_fan(report["bootstrap"], result.trades["return_pct"])
    assert [t.name for t in fig.data] == ["5–95%", "25–75%", "Median", "Realised"]


def test_fan_bands_are_one_hue_at_two_depths(scenario):
    """Confidence is a quantity, not an identity — it takes depth, not a second hue."""
    result, _, report = scenario
    fig = charts.build_bootstrap_fan(report["bootstrap"], result.trades["return_pct"])
    fills = [t.fillcolor for t in fig.data if t.fill == "toself"]
    assert len(fills) == 2
    channels = {f.rsplit(",", 1)[0] for f in fills}
    assert len(channels) == 1, "bands must share a hue and differ only in alpha"


def test_fan_realised_path_uses_a_contrasting_slot(scenario):
    result, _, report = scenario
    fig = charts.build_bootstrap_fan(report["bootstrap"], result.trades["return_pct"])
    realised = [t for t in fig.data if t.name == "Realised"][0]
    assert realised.line.color == viz.CATEGORICAL[1]


def test_fan_starts_every_path_at_one(scenario):
    result, _, report = scenario
    fig = charts.build_bootstrap_fan(report["bootstrap"], result.trades["return_pct"])
    median = [t for t in fig.data if t.name == "Median"][0]
    assert float(median.y[0]) == pytest.approx(1.0)


def test_fan_is_labelled_as_per_trade_not_portfolio_equity(scenario):
    """Regression: both charts once said "Growth of 1.0" while measuring
    different things — the fan compounds trades at a constant stake, the equity
    chart holds cash between them."""
    result, _, report = scenario
    fan = charts.build_bootstrap_fan(report["bootstrap"], result.trades["return_pct"])
    equity = charts.build_equity_comparison(result, report["buy_and_hold"])
    assert "staked" in fan.layout.yaxis.title.text
    assert "Portfolio equity" in equity.layout.yaxis.title.text
    assert fan.layout.yaxis.title.text != equity.layout.yaxis.title.text


def test_fan_without_a_realised_path_still_draws(scenario):
    _, _, report = scenario
    fig = charts.build_bootstrap_fan(report["bootstrap"])
    assert "Realised" not in [t.name for t in fig.data]


# ---------------------------------------------------------------------------
# Random entry
# ---------------------------------------------------------------------------

def test_random_entry_marks_the_strategy_with_a_direct_label(scenario):
    """The percentile is in the annotation, so colour never carries it alone."""
    _, _, report = scenario
    fig = charts.build_random_entry_distribution(report["random_entry"])
    labels = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "Strategy" in labels
    assert "pct" in labels


def test_random_entry_plots_every_simulated_path(scenario):
    _, _, report = scenario
    bench = report["random_entry"]
    fig = charts.build_random_entry_distribution(bench)
    assert len(fig.data[0].x) == bench["n_paths"]


# ---------------------------------------------------------------------------
# Per symbol
# ---------------------------------------------------------------------------

def test_per_symbol_pairs_traded_against_held(scenario):
    result, universe, report = scenario
    fig = charts.build_per_symbol_returns(result, report["buy_and_hold"])
    assert [t.name for t in fig.data] == ["Strategy", "Buy & hold"]
    assert fig.layout.barmode == "group"
    assert set(fig.data[0].x) <= set(universe)


def test_per_symbol_separates_bars_with_the_surface(scenario):
    result, _, report = scenario
    fig = charts.build_per_symbol_returns(result, report["buy_and_hold"])
    for trace in fig.data:
        assert trace.marker.line.color == viz.SURFACE
        assert trace.marker.line.width == 2


def test_per_symbol_handles_a_name_that_was_never_traded(scenario):
    result, _, report = scenario
    buy_hold = dict(report["buy_and_hold"])
    buy_hold["per_symbol"] = {**buy_hold["per_symbol"], "ZZZ": 0.25}
    fig = charts.build_per_symbol_returns(result, buy_hold)
    assert "ZZZ" in list(fig.data[0].x)


# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------

def test_rgba_preserves_the_hue_and_applies_alpha():
    assert viz.rgba("#3987e5", 0.5) == "rgba(57,135,229,0.5)"


def test_swing_charts_and_backtest_charts_share_one_theme():
    import swing_charts
    assert swing_charts.SURFACE is viz.SURFACE
    assert swing_charts.CATEGORICAL is viz.CATEGORICAL


# ---------------------------------------------------------------------------
# Swing / bracket-strategy charts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def swing_scenario():
    import swing_backtest as sb
    import swing_screener as ss

    bars = ss.load_demo(n_names=4, seed=11)
    spy = bars.pop("SPY")
    config = sb.SwingBacktestConfig(warmup=430)
    result = sb.run_swing_backtest(bars, spy, dict(ss.CFG), config)
    if result.trades.empty:
        pytest.skip("swing run produced no trades")
    return result, bars, spy, config


def test_order_funnel_narrows_from_setups_to_fills(swing_scenario):
    result, *_ = swing_scenario
    fig = charts.build_order_funnel(result.stats, len(result.trades))
    counts = list(fig.data[0].x)
    assert counts == sorted(counts), "the funnel is drawn bottom-up and cannot widen"


def test_outcome_breakdown_counts_unfilled_orders_from_the_stats(swing_scenario):
    """An unfilled order is not a trade, so it cannot come from the trade table."""
    result, *_ = swing_scenario
    fig = charts.build_outcome_breakdown(result.trades, result.stats)
    labels = list(fig.data[0].y)
    values = dict(zip(labels, fig.data[0].x))
    assert "Never filled" in labels
    assert values["Never filled"] == result.stats["orders_expired"]


def test_outcome_breakdown_uses_the_engines_exit_reasons(swing_scenario):
    result, *_ = swing_scenario
    keys = {k for k, _ in charts.OUTCOME_ORDER if not k.startswith("__")}
    assert set(result.trades["exit_reason"]) <= keys


def test_outcome_breakdown_without_stats_omits_the_unfilled_row(swing_scenario):
    result, *_ = swing_scenario
    fig = charts.build_outcome_breakdown(result.trades)
    assert "Never filled" not in list(fig.data[0].y)


def test_r_distribution_splits_at_zero_and_marks_expectancy(swing_scenario):
    result, *_ = swing_scenario
    fig = charts.build_r_multiple_distribution(result.trades)
    names = [t.name for t in fig.data]
    assert set(names) <= {"Losses", "Wins"}
    labels = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "Expectancy" in labels


def test_r_fan_accumulates_rather_than_compounds(swing_scenario):
    """R is additive: twenty 1R wins is +20R, not 1.01**20."""
    import simulation as sim
    result, *_ = swing_scenario
    paths = sim.bootstrap_r_paths(result.trades["r_multiple"],
                                  sim.SimulationParams(n_paths=100, seed=3))
    fig = charts.build_r_fan(paths, result.trades["r_multiple"])
    realised = [t for t in fig.data if t.name == "Realised"][0]
    expected = float(result.trades["r_multiple"].sum())
    assert float(realised.y[-1]) == pytest.approx(expected, abs=1e-9)
    assert float(realised.y[0]) == pytest.approx(0.0)


def test_mae_chart_marks_the_stop_distance(swing_scenario):
    result, *_ = swing_scenario
    fig = charts.build_mae_vs_outcome(result.trades)
    labels = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "stop distance" in labels
    assert len(fig.data[0].x) == len(result.trades)


def test_ambiguity_chart_shows_both_resolutions(swing_scenario):
    import swing_backtest as sb
    import swing_screener as ss
    _, bars, spy, config = swing_scenario
    bound = sb.ambiguity_bound(bars, spy, dict(ss.CFG), config)
    fig = charts.build_ambiguity_bound(bound)
    names = [t.name for t in fig.data]
    assert any("optimistic" in n for n in names)
    assert any("pessimistic" in n for n in names)


def test_swing_charts_share_the_theme(swing_scenario):
    import simulation as sim
    result, bars, spy, config = swing_scenario
    paths = sim.bootstrap_r_paths(result.trades["r_multiple"],
                                  sim.SimulationParams(n_paths=80, seed=3))
    figures = [
        charts.build_order_funnel(result.stats, len(result.trades)),
        charts.build_outcome_breakdown(result.trades, result.stats),
        charts.build_r_multiple_distribution(result.trades),
        charts.build_r_fan(paths, result.trades["r_multiple"]),
        charts.build_mae_vs_outcome(result.trades),
    ]
    for fig in figures:
        assert fig.layout.paper_bgcolor == viz.SURFACE
        assert fig.layout.xaxis.griddash == "solid"
        assert "yaxis2" not in fig.layout
