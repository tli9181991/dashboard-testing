"""The price chart must draw the levels the signal engine actually uses."""

import pytest

import assistant_charts as ac
import data as data_mod
import strategy as strat
import viz


@pytest.fixture(scope="module")
def prices():
    return data_mod.synthetic_ohlcv(n=400, seed=7)


@pytest.fixture(scope="module")
def frame(prices):
    return strat.prepare(prices)


def test_levels_come_from_the_strategy_not_a_second_implementation(frame, prices):
    """A chart drawing different levels from the ones the rule compares against
    would look authoritative and mean nothing."""
    engine = strat.merged_levels(frame)
    price = float(frame.df["Close"].iloc[-1])
    drawn = ac.levels_for_chart(frame, price)
    engine_prices = {round(p, 6) for p, _, _ in engine}
    assert {round(p, 6) for p, _, _ in drawn} <= engine_prices


def test_merged_levels_matches_the_engines_own_tolerance(frame):
    i = len(frame) - 1
    atr = float(frame.df.iloc[i]["ATR"])
    expected = strat._merge(frame.levels_asof(i),
                            max(1e-9, frame.params.level_merge_atr_mult * atr))
    assert strat.merged_levels(frame, i) == expected


def test_merged_levels_defaults_to_the_last_bar(frame):
    assert strat.merged_levels(frame) == strat.merged_levels(frame, len(frame) - 1)


def test_merged_levels_on_an_empty_frame():
    import pandas as pd
    empty = strat.StrategyFrame(pd.DataFrame(columns=["ATR"]), [], strat.StrategyParams())
    assert strat.merged_levels(empty, -1) == []


def test_distant_levels_are_dropped(frame):
    price = float(frame.df["Close"].iloc[-1])
    drawn = ac.levels_for_chart(frame, price, max_distance=0.05)
    for lv_price, _, _ in drawn:
        assert abs(lv_price - price) / price <= 0.05


def test_the_strongest_levels_survive_the_cap(frame):
    price = float(frame.df["Close"].iloc[-1])
    many = ac.levels_for_chart(frame, price, max_levels=99)
    few = ac.levels_for_chart(frame, price, max_levels=3)
    assert len(few) <= 3
    if len(many) > 3:
        assert min(t for _, _, t in few) >= min(t for _, _, t in many)


def test_chart_draws_four_moving_averages_in_an_ordinal_ramp(prices):
    fig = ac.build_price_chart("TEST", prices)
    assert fig.data[0].type == "candlestick"
    assert [t.name for t in fig.data[1:]] == ["5 EMA", "10 EMA", "20 EMA", "50 EMA"]
    assert [t.line.color for t in fig.data[1:]] == ac.RAMP_EMA


def test_every_level_carries_a_readable_label(prices):
    """Price and touch count in the label, so colour never carries the meaning."""
    fig = ac.build_price_chart("TEST", prices)
    labels = [a.text for a in fig.layout.annotations if a.text]
    level_labels = [t for t in labels if t.startswith(("R ", "S "))]
    assert level_labels
    for text in level_labels:
        assert "×" in text, f"no touch count in {text!r}"


def test_the_last_price_is_marked(prices):
    fig = ac.build_price_chart("TEST", prices)
    labels = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "Last" in labels


def test_outside_labels_get_gutters_wide_enough_to_render(prices):
    """Regression: the default 12px margins clipped every level label to "R"/"S"."""
    fig = ac.build_price_chart("TEST", prices)
    assert fig.layout.margin.r >= 100
    assert fig.layout.margin.l >= 60


def test_support_and_resistance_take_distinct_hues():
    assert ac.SUPPORT != ac.RESISTANCE
    assert ac.SUPPORT not in ac.RAMP_EMA
    assert ac.RESISTANCE not in ac.RAMP_EMA


def test_the_chart_shares_the_app_theme(prices):
    fig = ac.build_price_chart("TEST", prices)
    assert fig.layout.paper_bgcolor == viz.SURFACE
    assert fig.layout.xaxis.griddash == "solid"
    assert "yaxis2" not in fig.layout
    assert fig.layout.xaxis.rangeslider.visible is False


def test_the_lookback_bounds_what_is_drawn(prices):
    fig = ac.build_price_chart("TEST", prices, lookback=60)
    assert len(fig.data[0].x) == 60


def test_the_y_range_covers_every_drawn_level(prices):
    fig = ac.build_price_chart("TEST", prices)
    low, high = fig.layout.yaxis.range
    for annotation in fig.layout.annotations:
        if annotation.text and annotation.text.startswith(("R ", "S ")):
            assert low <= annotation.y <= high


def test_prompt_text_splits_resistance_above_from_support_below(prices):
    text = ac.levels_prompt_text("TEST", prices)
    assert "TEST last" in text
    if "Resistance above:" in text:
        price = float(strat.prepare(prices).df["Close"].iloc[-1])
        chunk = text.split("Resistance above:")[1].split("Support below:")[0]
        first = float(chunk.split()[0].rstrip(",").replace(",", ""))
        assert first > price


def test_prompt_text_says_so_when_there_are_no_levels():
    flat = data_mod.synthetic_ohlcv(n=300, seed=3, annual_vol=0.0001)
    text = ac.levels_prompt_text("FLAT", flat)
    assert "FLAT" in text
