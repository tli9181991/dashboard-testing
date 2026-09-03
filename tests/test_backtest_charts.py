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
