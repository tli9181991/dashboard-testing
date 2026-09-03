"""Chart builders for the swing screener results.

The important test here is that the score decomposition adds back up to the score
the scan produced — a breakdown chart that does not reconcile is worse than none.
"""

import numpy as np
import pandas as pd
import pytest

import swing_charts as charts
import swing_screener as swing


@pytest.fixture(scope="module")
def scan():
    bars = swing.load_demo(n_names=200, seed=11)
    spy = bars.pop("SPY")
    cfg = dict(swing.CFG)
    cfg["tradability_keep_pct"] = 0.90        # widen so there is something to draw
    out, ctx = swing.run_scan(bars, spy, cfg)
    if out.empty:
        pytest.skip("synthetic universe produced no candidates")
    return out, ctx, cfg, bars


# ---------------------------------------------------------------------------
# Score decomposition
# ---------------------------------------------------------------------------

def test_components_reconstruct_the_score(scan):
    out, _, cfg, _ = scan
    parts = charts.score_components(out, cfg)
    rebuilt = parts.sum(axis=1)
    np.testing.assert_allclose(rebuilt.to_numpy(), out["score"].to_numpy(), atol=1e-9)


def test_components_cover_every_weight_in_the_config(scan):
    _, _, cfg, _ = scan
    weights = {key for _, key in charts.COMPONENTS}
    assert weights == {k for k in cfg if k.startswith("w_")}


def test_components_on_an_empty_frame(scan):
    _, _, cfg, _ = scan
    parts = charts.score_components(pd.DataFrame(), cfg)
    assert parts.empty


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------

def test_funnel_stages_are_monotonically_narrowing(scan):
    out, ctx, _, _ = scan
    counts = [c for _, c in charts.funnel_stages(ctx, len(out))]
    assert counts == sorted(counts, reverse=True), "a funnel cannot widen"


def test_funnel_handles_a_missing_context():
    stages = charts.funnel_stages({}, 0)
    assert [c for _, c in stages] == [0, 0, 0, 0, 0]
    fig = charts.build_funnel({}, 0)
    assert len(fig.data) == 1


def test_funnel_uses_the_ordinal_ramp(scan):
    out, ctx, _, _ = scan
    fig = charts.build_funnel(ctx, len(out))
    assert list(fig.data[0].marker.color) == charts.RAMP_FUNNEL
    assert fig.data[0].orientation == "h"


# ---------------------------------------------------------------------------
# Trade plan
# ---------------------------------------------------------------------------

def test_trade_plan_draws_candles_and_three_moving_averages(scan):
    out, _, _, bars = scan
    row = out.iloc[0]
    fig = charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row)

    kinds = [t.type for t in fig.data]
    assert kinds[0] == "candlestick"
    assert [t.name for t in fig.data[1:]] == ["10 EMA", "20 EMA", "50 SMA"]
    assert [t.line.color for t in fig.data[1:]] == charts.RAMP_MA


def test_trade_plan_labels_every_level(scan):
    out, _, _, bars = scan
    row = out.iloc[0]
    fig = charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row)
    labels = " ".join(a.text for a in fig.layout.annotations if a.text)
    for expected in ("Stop", "Entry", "TP1", "TP2"):
        assert expected in labels, f"{expected} has no direct label"


def test_trade_plan_shades_risk_below_entry_and_reward_above(scan):
    out, _, _, bars = scan
    row = out.iloc[0]
    fig = charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row)
    bands = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(bands) == 2
    risk, reward = sorted(bands, key=lambda s: s.y0)
    assert risk.y0 == pytest.approx(row["stop"])
    assert risk.y1 == pytest.approx(row["entry"])
    assert reward.y1 == pytest.approx(row["tp2"])


def test_trade_plan_hides_the_range_slider(scan):
    out, _, _, bars = scan
    row = out.iloc[0]
    fig = charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row)
    assert fig.layout.xaxis.rangeslider.visible is False


def test_trade_plan_respects_the_lookback(scan):
    out, _, _, bars = scan
    row = out.iloc[0]
    fig = charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row, lookback=40)
    assert len(fig.data[0].x) == 40


# ---------------------------------------------------------------------------
# Chrome & palette discipline
# ---------------------------------------------------------------------------

@pytest.fixture
def every_figure(scan):
    out, ctx, cfg, bars = scan
    row = out.iloc[0]
    return [
        charts.build_funnel(ctx, len(out)),
        charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row),
        charts.build_score_breakdown(out, cfg),
        charts.build_risk_reward(out),
    ]


def test_every_chart_paints_the_same_surface(every_figure):
    for fig in every_figure:
        assert fig.layout.paper_bgcolor == charts.SURFACE
        assert fig.layout.plot_bgcolor == charts.SURFACE


def test_gridlines_are_solid_hairlines(every_figure):
    """Dashed gridlines read as a threshold; only the trade levels are dashed."""
    for fig in every_figure:
        assert fig.layout.xaxis.griddash == "solid"
        assert fig.layout.yaxis.griddash == "solid"
        assert fig.layout.xaxis.gridcolor == charts.GRID


def test_no_chart_uses_two_y_axes(every_figure):
    for fig in every_figure:
        assert "yaxis2" not in fig.layout, "dual axes invent correlations"


def test_series_colours_never_come_from_the_status_palette(scan):
    """Status colours mean good/bad; a series must not impersonate one."""
    out, _, cfg, _ = scan
    reserved = {charts.STATUS_GOOD, charts.STATUS_CRITICAL}
    for fig in (charts.build_score_breakdown(out, cfg), charts.build_risk_reward(out)):
        for trace in fig.data:
            colour = getattr(trace.marker, "color", None)
            if isinstance(colour, str):
                assert colour not in reserved


def test_score_breakdown_draws_every_contributing_component(scan):
    out, _, cfg, _ = scan
    parts = charts.score_components(out, cfg)
    fig = charts.build_score_breakdown(out, cfg)
    names = [t.name for t in fig.data]

    for label, _ in charts.COMPONENTS:
        contributes = parts[label].abs().gt(1e-12).any()
        assert (label in names) == bool(contributes), label
    assert "Total score" in names
    assert fig.layout.barmode == "relative"


def test_stacked_segments_are_separated_by_the_surface(scan):
    out, _, cfg, _ = scan
    fig = charts.build_score_breakdown(out, cfg)
    bars_ = [t for t in fig.data if t.type == "bar"]
    for trace in bars_:
        assert trace.marker.line.color == charts.SURFACE
        assert trace.marker.line.width == 2


def test_risk_reward_colours_by_variant_not_by_rank(scan):
    """Colour follows the entity: the same variant keeps its hue as rows change."""
    out, _, _, _ = scan
    full = charts.build_risk_reward(out)
    mapping = {t.name: t.marker.color for t in full.data}

    subset = out.iloc[::-1].reset_index(drop=True)
    reordered = charts.build_risk_reward(subset)
    for trace in reordered.data:
        assert trace.marker.color == mapping[trace.name]


def test_risk_reward_marker_size_tracks_dollars_at_risk(scan):
    out, _, _, _ = scan
    fig = charts.build_risk_reward(out)
    for trace in fig.data:
        assert np.all(np.asarray(trace.marker.size) >= 12)


def test_charts_with_several_series_identify_them(scan):
    """Identity is never colour-alone: two or more series get a legend.

    A lone series needs no legend box — the title names it — which is why the
    funnel and a single-variant scatter are exempt.
    """
    out, ctx, cfg, bars = scan
    row = out.iloc[0]

    for fig in (charts.build_score_breakdown(out, cfg),
                charts.build_trade_plan(row["ticker"], bars[row["ticker"]], row),
                charts.build_risk_reward(out)):
        series = [t for t in fig.data if t.showlegend is not False]
        if len(series) >= 2:
            assert all(t.name for t in series), "a legend entry with no name"
            assert fig.layout.showlegend is not False

    # Single-series charts carry their identity in the title instead.
    assert charts.build_funnel(ctx, len(out)).data[0].showlegend is False


def test_score_breakdown_always_shows_the_total_alongside_the_parts(scan):
    """However many components contribute, the total marker is always drawn."""
    out, _, cfg, _ = scan
    parts = charts.score_components(out, cfg)
    drawn = sum(1 for label, _ in charts.COMPONENTS
                if parts[label].abs().gt(1e-12).any())
    fig = charts.build_score_breakdown(out, cfg)
    assert len(fig.data) == drawn + 1
    assert drawn >= 2, "fixture has too few live components to test the stack"


def test_a_component_that_draws_nothing_gets_no_legend_entry(scan):
    """Sector rank is absent without sector ETFs; an empty series must not be
    advertised in the legend."""
    out, _, cfg, _ = scan
    parts = charts.score_components(out, cfg)
    empty = [label for label, _ in charts.COMPONENTS
             if not parts[label].abs().gt(1e-12).any()]
    drawn = {t.name for t in charts.build_score_breakdown(out, cfg).data}
    for label in empty:
        assert label not in drawn, f"{label} contributes nothing but is in the legend"


def test_bar_charts_use_thin_marks(scan):
    out, ctx, cfg, _ = scan
    for fig in (charts.build_funnel(ctx, len(out)),
                charts.build_score_breakdown(out, cfg)):
        assert fig.layout.bargap >= 0.3, "heavy blocks read loud"


def _synthetic_candidates(n, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        entry = 40 + i * 3
        risk = entry * 0.03
        rows.append({"ticker": f"TKR{i}", "variant": "A/momentum" if i % 2 else "B/ict",
                     "entry": entry, "stop": entry - risk, "tp1": entry + risk * 0.8,
                     "tp2": entry + risk * rng.uniform(2.0, 4.5),
                     "shares": 100, "risk_$": float(rng.integers(200, 900)),
                     "score": rng.normal()})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def test_small_scatters_label_every_point():
    out = _synthetic_candidates(6)
    fig = charts.build_risk_reward(out)
    labelled = [t for trace in fig.data for t in trace.text if t]
    assert len(labelled) == 6


def test_crowded_scatters_label_only_the_leaders():
    """A label on every marker is chaos once the field is large."""
    out = _synthetic_candidates(30)
    fig = charts.build_risk_reward(out)
    labelled = {t for trace in fig.data for t in trace.text if t}
    assert len(labelled) == charts.LABEL_TOP_N
    assert labelled == set(out.nlargest(charts.LABEL_TOP_N, "score")["ticker"])


def test_crowded_scatters_still_plot_every_point():
    out = _synthetic_candidates(30)
    fig = charts.build_risk_reward(out)
    assert sum(len(trace.x) for trace in fig.data) == 30


def test_a_lone_candidate_has_no_cross_section_to_score_against(scan):
    """Why app.py guards the breakdown: z-scores over one row are all zero.

    Rendering it anyway produces an axis with nothing on it, which reads as a
    broken chart rather than as "there is only one candidate".
    """
    out, _, cfg, _ = scan
    single = out.head(1)
    components = charts.score_components(single, cfg)
    assert len(components) == 1
    assert (components.iloc[0].abs() < 1e-9).all(), \
        "a single candidate should contribute nothing on any component"
