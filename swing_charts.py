"""Plotly figures for the Swing Universe Funnel results.

Pure builders — each takes data and returns a ``go.Figure``, so they can be tested
without Streamlit.

Colour assignments follow the job each encoding does, not taste:

* **Funnel stages** are ordered categories, so they take a single-hue *ordinal*
  blue ramp, light to dark.
* **Moving averages** are also ordered (10 → 20 → 50 sessions), so they take the
  same kind of ramp rather than three unrelated hues.
* **Score components** are identities with no order, so they take the categorical
  slots in fixed order.
* **Entry, stop and targets** are states, so they take reserved status colours —
  and every one carries a direct text label, so the colour never has to carry the
  meaning alone.

Every palette here was checked with the data-viz validator against the dark chart
surface (#1a1a19): the categorical four and the two-slot scatter pass the
adjacent and all-pairs gates respectively; both ordinal ramps pass monotonicity,
step separation and light-end contrast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Surface & ink ────────────────────────────────────────────────────────────
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

# ── Status (reserved; never used for a series) ───────────────────────────────
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# ── Categorical slots, fixed order (dark steps) ──────────────────────────────
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500"]

# ── Ordinal blue ramps, light → dark ─────────────────────────────────────────
RAMP_MA = ["#9ec5f4", "#5598e7", "#256abf"]
RAMP_FUNNEL = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(fig: go.Figure, height: int, title: str = "") -> go.Figure:
    """Recessive chrome: hairline solid grid, muted ink, generous padding."""
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=14, color=INK_PRIMARY)) if title else None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        margin=dict(l=12, r=12, t=42 if title else 16, b=12),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(family=FONT, color=INK_PRIMARY),
                        bordercolor=AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    font=dict(color=INK_SECONDARY)),
    )
    fig.update_xaxes(gridcolor=GRID, griddash="solid", zeroline=False,
                     linecolor=AXIS, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, griddash="solid", zeroline=False,
                     linecolor=AXIS, tickfont=dict(color=INK_MUTED))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Funnel — how the universe narrows
# ─────────────────────────────────────────────────────────────────────────────

def funnel_stages(ctx: dict, n_candidates: int) -> list[tuple[str, int]]:
    return [
        ("Universe", int(ctx.get("universe", 0) or 0)),
        ("Liquidity gate", int(ctx.get("gated", 0) or 0)),
        ("Tradability floor", int(ctx.get("floor_ok", 0) or 0)),
        ("Tradable", int(ctx.get("tradable", 0) or 0)),
        ("With a setup", int(n_candidates)),
    ]


def build_funnel(ctx: dict, n_candidates: int) -> go.Figure:
    """Ordered stages, one measure — an ordinal ramp, not eight hues."""
    stages = funnel_stages(ctx, n_candidates)
    labels = [s for s, _ in stages]
    counts = [c for _, c in stages]
    start = counts[0] or 1
    share = [f"{c / start:.0%} of universe" for c in counts]

    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color=RAMP_FUNNEL, line=dict(color=SURFACE, width=2)),
        text=[f"{c:,}" for c in counts],
        textposition="outside",
        textfont=dict(color=INK_SECONDARY),
        customdata=share,
        hovertemplate="<b>%{y}</b><br>%{x:,} symbols<br>%{customdata}<extra></extra>",
        showlegend=False,
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, rangemode="tozero")
    _base_layout(fig, height=260, title="Funnel — where the universe drops out")
    # Thin marks: saturated fills belong on small marks, not heavy blocks.
    fig.update_layout(margin=dict(l=12, r=60, t=42, b=12), bargap=0.45)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trade plan — the setup in its own price context
# ─────────────────────────────────────────────────────────────────────────────

def build_trade_plan(ticker: str, bars: pd.DataFrame, candidate: pd.Series,
                     lookback: int = 120) -> go.Figure:
    """Candlesticks with the entry, stop and targets drawn on the price scale.

    The risk band (stop → entry) and reward band (entry → TP2) are shaded so the
    geometry of the trade is visible without arithmetic.
    """
    df = bars.tail(lookback)
    entry = float(candidate["entry"])
    stop = float(candidate["stop"])
    tp1 = float(candidate["tp1"])
    tp2 = float(candidate["tp2"])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, showlegend=False,
        increasing=dict(line=dict(color=STATUS_GOOD, width=1), fillcolor=STATUS_GOOD),
        decreasing=dict(line=dict(color=STATUS_CRITICAL, width=1), fillcolor=STATUS_CRITICAL),
    ))

    close = df["Close"]
    for (label, series), colour in zip(
        (("10 EMA", close.ewm(span=10, adjust=False).mean()),
         ("20 EMA", close.ewm(span=20, adjust=False).mean()),
         ("50 SMA", close.rolling(50).mean())),
        RAMP_MA,
    ):
        fig.add_trace(go.Scatter(
            x=df.index, y=series, mode="lines", name=label,
            line=dict(color=colour, width=2),
            hovertemplate=f"{label} %{{y:$,.2f}}<extra></extra>",
        ))

    # Bands first, so the candles sit on top of them.
    fig.add_hrect(y0=stop, y1=entry, fillcolor=STATUS_CRITICAL, opacity=0.10,
                  line_width=0, layer="below")
    fig.add_hrect(y0=entry, y1=tp2, fillcolor=STATUS_GOOD, opacity=0.08,
                  line_width=0, layer="below")

    for value, label, colour, dash in (
        (stop, f"Stop ${stop:,.2f}", STATUS_CRITICAL, "dash"),
        (entry, f"Entry ${entry:,.2f}", INK_PRIMARY, "dash"),
        (tp1, f"TP1 ${tp1:,.2f}", STATUS_GOOD, "dot"),
        (tp2, f"TP2 ${tp2:,.2f}", STATUS_GOOD, "dash"),
    ):
        fig.add_hline(y=value, line=dict(color=colour, width=1.5, dash=dash),
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(color=colour, size=11))

    risk = entry - stop
    reward = tp2 - entry
    subtitle = (f"{ticker} · {candidate['variant']} · risk ${risk:,.2f}/share · "
                f"{reward / risk:.1f}R to TP2" if risk > 0 else ticker)

    _base_layout(fig, height=460, title=subtitle)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      margin=dict(l=12, r=110, t=42, b=12))
    fig.update_yaxes(title_text="Price ($)", title_font=dict(color=INK_MUTED))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Score decomposition — why a candidate ranked where it did
# ─────────────────────────────────────────────────────────────────────────────

COMPONENTS = (
    ("Tradability", "w_tradability"),
    ("Setup quality", "w_setup"),
    ("Relative strength", "w_rs"),
    ("Sector", "w_sector"),
)


def score_components(out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Reproduce the weighted terms that ``run_scan`` summed into ``score``.

    The scan keeps only the total, so the parts are recomputed here from the same
    columns with the same z-scoring. ``test_swing_charts`` asserts they add back
    up to the stored score.
    """
    import swing_screener as swing

    if out.empty:
        return pd.DataFrame(columns=[name for name, _ in COMPONENTS])

    ranks = out["sector_rank"]
    filler = ranks.mean() if ranks.notna().any() else 0
    sector_term = swing.zscore(-ranks.fillna(filler))

    parts = pd.DataFrame({
        "Tradability": cfg["w_tradability"] * swing.zscore(out["tradability"]),
        "Setup quality": cfg["w_setup"] * swing.zscore(out["quality"]),
        "Relative strength": cfg["w_rs"] * swing.zscore(out["rs_pct"]),
        "Sector": cfg["w_sector"] * sector_term,
    }, index=out.index)
    return parts


def build_score_breakdown(out: pd.DataFrame, cfg: dict, top_n: int = 10) -> go.Figure:
    """Stacked contributions per candidate. Segments can sit either side of zero:
    these are z-scores, so a component can drag a name down as well as lift it."""
    shown = out.head(top_n)
    parts = score_components(shown, cfg)
    tickers = shown["ticker"].tolist()

    fig = go.Figure()
    for (label, _), colour in zip(COMPONENTS, CATEGORICAL):
        column = parts[label]
        # A component can be identically zero — sector rank is unavailable when no
        # sector ETFs were loaded. Drawing nothing while keeping a legend entry
        # would advertise a series that is not on the chart.
        if not column.abs().gt(1e-12).any():
            continue
        fig.add_trace(go.Bar(
            y=tickers, x=column, orientation="h", name=label,
            marker=dict(color=colour, line=dict(color=SURFACE, width=2)),
            hovertemplate=f"<b>%{{y}}</b><br>{label} %{{x:+.3f}}<extra></extra>",
        ))

    # The stack runs both ways from zero, so the algebraic total has no visual
    # edge to read. This tick is where it actually lands.
    fig.add_trace(go.Scatter(
        y=tickers, x=shown["score"], mode="markers", name="Total score",
        marker=dict(symbol="line-ns", size=18,
                    line=dict(color=INK_PRIMARY, width=3)),
        hovertemplate="<b>%{y}</b><br>Total %{x:+.3f}<extra></extra>",
    ))

    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinecolor=AXIS, zerolinewidth=1,
                     title_text="Weighted contribution to composite score",
                     title_font=dict(color=INK_MUTED))
    _base_layout(fig, height=max(240, 46 * len(tickers) + 90),
                 title="What drove each ranking")
    fig.update_layout(barmode="relative", bargap=0.45,
                      margin=dict(l=12, r=32, t=42, b=12))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Risk / reward — payoff against rank
# ─────────────────────────────────────────────────────────────────────────────

#: Past this many points, a label on every marker is noise, so only the leaders
#: are named and the rest are left to the axis and the tooltip.
LABEL_ALL_UP_TO = 12
LABEL_TOP_N = 5


def build_risk_reward(out: pd.DataFrame, label_all_up_to: int = LABEL_ALL_UP_TO) -> go.Figure:
    """Reward multiple against composite score, sized by dollars at risk."""
    plot = out.copy()
    risk_per_share = (plot["entry"] - plot["stop"]).replace(0, np.nan)
    plot["r_tp2"] = (plot["tp2"] - plot["entry"]) / risk_per_share
    plot = plot.dropna(subset=["r_tp2"])

    # Direct-label selectively: every point when the field is small, otherwise
    # just the highest-scoring few.
    if len(plot) <= label_all_up_to:
        named = set(plot["ticker"])
    else:
        named = set(plot.nlargest(LABEL_TOP_N, "score")["ticker"])
    plot["label"] = [t if t in named else "" for t in plot["ticker"]]

    max_risk = float(plot["risk_$"].max()) or 1.0
    fig = go.Figure()
    for variant, colour in zip(sorted(plot["variant"].unique()), CATEGORICAL):
        grp = plot[plot["variant"] == variant]
        fig.add_trace(go.Scatter(
            x=grp["r_tp2"], y=grp["score"], mode="markers+text", name=variant,
            text=grp["label"], textposition="top center",
            textfont=dict(color=INK_MUTED, size=10),
            marker=dict(
                color=colour, size=12 + 22 * (grp["risk_$"] / max_risk),
                line=dict(color=SURFACE, width=2),
            ),
            customdata=np.stack([grp["ticker"], grp["risk_$"], grp["shares"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b><br>%{x:.1f}R to TP2<br>"
                           "score %{y:+.2f}<br>$%{customdata[1]:,.0f} at risk"
                           " · %{customdata[2]:,} shares<extra></extra>"),
        ))

    fig.update_xaxes(title_text="Reward multiple to TP2 (R)", title_font=dict(color=INK_MUTED))
    fig.update_yaxes(title_text="Composite score", title_font=dict(color=INK_MUTED))
    _base_layout(fig, height=420, title="Payoff against rank — marker size is dollars at risk")
    return fig
