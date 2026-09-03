"""Plotly figures for a backtest and its Monte Carlo layer.

Pure builders — each takes data and returns a ``go.Figure``, so they can be tested
without Streamlit.

Colour by the job each encoding does:

* **Strategy vs buy-and-hold** are two identities with no order, so they take
  categorical slots in fixed order.
* **Bootstrap percentile bands** are one quantity at different confidence, so they
  are one hue at two depths, not two hues.
* **The realised path and the strategy's actual average trade** are the reference
  the rest is being judged against, so they take a contrasting categorical slot and
  a direct label — never colour alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz import (
    CATEGORICAL,
    INK_MUTED,
    INK_PRIMARY,
    RAMP_FUNNEL,
    STATUS_CRITICAL,
    STATUS_GOOD,
    SURFACE,
    base_layout,
    rgba,
)

BAND_HUE = RAMP_FUNNEL[3]          # mid blue; bands are one hue at two depths
REALISED = CATEGORICAL[1]          # orange, so it reads against the blue bands


def build_equity_comparison(result, buy_hold: dict, height: int = 360) -> go.Figure:
    """Strategy equity against equal-weight buy-and-hold, both indexed to 1.0.

    The floor question: did trading beat not trading, on the same names over the
    same window?
    """
    fig = go.Figure()

    equity = result.equity
    start = float(equity.iloc[0]) if len(equity) else 1.0
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity / start if start else equity,
        mode="lines", name="Strategy",
        line=dict(color=CATEGORICAL[0], width=2),
        hovertemplate="Strategy %{y:.2f}×<extra></extra>",
    ))

    curve = (buy_hold or {}).get("curve")
    if curve is not None and len(curve):
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve, mode="lines", name="Buy & hold",
            line=dict(color=CATEGORICAL[1], width=2),
            hovertemplate="Buy & hold %{y:.2f}×<extra></extra>",
        ))

    base_layout(fig, height, "Strategy against buy and hold")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="Portfolio equity (× starting)")
    return fig


def build_bootstrap_fan(path_result, realised_returns=None, height: int = 380) -> go.Figure:
    """Where the realised run sits among the sequences it could have produced.

    Bands are the 5–95 and 25–75 percentiles of resampled trade orders. A realised
    path near the edge of the fan means the outcome leaned heavily on the order the
    trades happened to arrive in.

    This compounds *per-trade* returns at a constant stake, so it will not match the
    portfolio equity curve, which also holds cash between trades and sizes each
    position to a fraction of the book. The two answer different questions and are
    labelled so they cannot be read as the same line.
    """
    paths = path_result.paths
    x = np.arange(paths.shape[1])

    lo90, hi90 = path_result.percentile_band(5, 95)
    lo50, hi50 = path_result.percentile_band(25, 75)

    fig = go.Figure()
    for lo, hi, alpha, label in ((lo90, hi90, 0.16, "5–95%"), (lo50, hi50, 0.30, "25–75%")):
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]), y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=rgba(BAND_HUE, alpha),
            line=dict(width=0), name=label, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=x, y=path_result.median_path, mode="lines", name="Median",
        line=dict(color=BAND_HUE, width=2),
        hovertemplate="Trade %{x} · median %{y:.2f}× staked<extra></extra>",
    ))

    if realised_returns is not None:
        series = pd.Series(realised_returns).dropna().to_numpy(dtype=float)
        if series.size:
            actual = np.concatenate([[1.0], np.cumprod(1.0 + series)])
            fig.add_trace(go.Scatter(
                x=np.arange(actual.size), y=actual, mode="lines", name="Realised",
                line=dict(color=REALISED, width=2.5),
                hovertemplate="Trade %{x} · realised %{y:.2f}× staked<extra></extra>",
            ))

    base_layout(fig, height, "Outcome range across resampled trade orders")
    fig.update_xaxes(title="Trade number")
    fig.update_yaxes(title="Growth of 1.0 staked per trade")
    fig.add_hline(y=1.0, line=dict(color=INK_MUTED, width=1))
    return fig


def build_random_entry_distribution(bench: dict, height: int = 320) -> go.Figure:
    """The signal test: the strategy's average trade against random entries with
    the same holding periods.

    Near the middle of the distribution means the entry rule added nothing over
    simply being in the market for the same amount of time.
    """
    draws = np.asarray(bench["distribution"], dtype=float)
    actual = float(bench["actual_mean_trade"])

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=draws, nbinsx=40, name="Random entries",
        marker=dict(color=rgba(BAND_HUE, 0.55), line=dict(width=0)),
        hovertemplate="Mean trade %{x:.2%}<br>%{y} paths<extra></extra>",
    ))

    fig.add_vline(
        x=actual, line=dict(color=REALISED, width=2.5),
        annotation_text=f"Strategy {actual:.2%} · {bench['percentile']:.0f}th pct",
        annotation_position="top",
        annotation=dict(font=dict(color=REALISED, size=12)),
    )

    base_layout(fig, height, "Strategy against random entries of the same length")
    fig.update_xaxes(title="Mean return per trade", tickformat=".1%")
    fig.update_yaxes(title="Simulated paths")
    fig.update_layout(bargap=0.02, showlegend=False)
    return fig


def build_per_symbol_returns(result, buy_hold: dict, height: int = 320) -> go.Figure:
    """Realised strategy PnL per name beside that name's buy-and-hold return.

    Two identities, so two categorical slots; a name where buy-and-hold wins is
    one the strategy traded worse than simply owning.
    """
    trades = result.trades
    per_symbol = (buy_hold or {}).get("per_symbol", {})
    symbols = sorted(set(per_symbol) | set(trades["symbol"].unique() if not trades.empty else []))
    if not symbols:
        symbols = sorted(per_symbol)

    strat = []
    for symbol in symbols:
        rows = trades[trades["symbol"] == symbol] if not trades.empty else pd.DataFrame()
        strat.append(float((1 + rows["return_pct"]).prod() - 1) if len(rows) else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols, y=strat, name="Strategy",
        marker=dict(color=CATEGORICAL[0], line=dict(width=2, color=SURFACE)),
        hovertemplate="%{x} strategy %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=symbols, y=[per_symbol.get(s, 0.0) for s in symbols], name="Buy & hold",
        marker=dict(color=CATEGORICAL[1], line=dict(width=2, color=SURFACE)),
        hovertemplate="%{x} buy & hold %{y:.1%}<extra></extra>",
    ))

    base_layout(fig, height, "Per name: traded against simply held")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title="Return", tickformat=".0%")
    fig.add_hline(y=0.0, line=dict(color=INK_MUTED, width=1))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Swing / bracket-strategy charts
#
# R multiples ADD rather than compound — every trade is sized to the same risk by
# construction — so anything cumulative here is a running sum, never a product.
# ─────────────────────────────────────────────────────────────────────────────

#: Exit reasons as swing_backtest records them, ordered worst to best.
OUTCOME_ORDER = [
    ("__unfilled__", "Never filled"),
    ("stop", "Stopped"),
    ("breakeven", "TP1 then breakeven"),
    ("tp1", "Reached TP1"),
    ("time_stop", "Time stop"),
    ("end_of_test", "Still open at the end"),
    ("tp2", "Reached TP2"),
]

def build_outcome_breakdown(trades: pd.DataFrame, stats: dict | None = None,
                            height: int = 300) -> go.Figure:
    """How the signals ended, worst to best.

    One series counting categories, so it takes a single ordinal ramp and direct
    labels rather than a hue per outcome. Orders that expired without trading are
    counted here from the order stats — they are the largest bucket for a bracket
    strategy and leaving them out makes the fill rate invisible.
    """
    counts = trades["exit_reason"].value_counts().to_dict() if not trades.empty else {}
    if stats:
        counts["__unfilled__"] = int(stats.get("orders_expired", 0))
    present = [(key, label) for key, label in OUTCOME_ORDER if counts.get(key)]
    if not present:
        present = [(key, label) for key, label in OUTCOME_ORDER[:2]]

    labels = [label for _, label in present]
    values = [counts.get(key, 0) for key, _ in present]
    ramp = RAMP_FUNNEL[-len(present):] if len(present) <= len(RAMP_FUNNEL) else \
        (RAMP_FUNNEL * len(present))[:len(present)]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=ramp, line=dict(width=0)),
        text=[f"{v:,}" for v in values], textposition="outside",
        cliponaxis=False, showlegend=False,
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    top = max(values) if values else 1
    base_layout(fig, height, "How the signals ended")
    fig.update_xaxes(range=[0, top * 1.18], title="Signals")
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.35)
    return fig

def build_order_funnel(stats: dict, n_trades: int, height: int = 240) -> go.Figure:
    """Setups seen → orders placed → orders actually filled.

    The gap between placed and filled is the whole reason a bracket strategy needs
    resting orders modelled: a setup whose entry never traded is not a trade.
    """
    stages = [
        ("Setups seen", int(stats.get("setups_seen", 0))),
        ("Orders placed", int(stats.get("orders_placed", 0))),
        ("Orders filled", int(n_trades)),
    ]
    labels = [s for s, _ in stages][::-1]
    counts = [c for _, c in stages][::-1]
    colours = [RAMP_FUNNEL[4], RAMP_FUNNEL[2], RAMP_FUNNEL[0]]

    fig = go.Figure(go.Bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{c:,}" for c in counts], textposition="outside",
        cliponaxis=False, showlegend=False,
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    base_layout(fig, height, "Setups that became positions")
    top = max(counts) if counts else 1
    fig.update_xaxes(range=[0, top * 1.18], title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(bargap=0.35)
    return fig

def build_r_multiple_distribution(trades: pd.DataFrame, height: int = 320) -> go.Figure:
    """Payoff shape in R. Split at zero because the sign is the point.

    A bracket strategy that takes partials shows many small wins against occasional
    full -1R losses; whether that nets out is what the expectancy tells you.
    """
    r = trades["r_multiple"].astype(float)
    fig = go.Figure()

    for subset, colour, name in ((r[r <= 0], STATUS_CRITICAL, "Losses"),
                                 (r[r > 0], STATUS_GOOD, "Wins")):
        if len(subset):
            fig.add_trace(go.Histogram(
                x=subset, xbins=dict(size=0.25), name=name,
                marker=dict(color=rgba(colour, 0.75), line=dict(width=0)),
                hovertemplate=name + " %{x:.2f}R · %{y} trades<extra></extra>",
            ))

    mean_r = float(r.mean())
    fig.add_vline(
        x=mean_r, line=dict(color=CATEGORICAL[0], width=2.5),
        annotation_text=f"Expectancy {mean_r:+.2f}R",
        annotation_position="top",
        annotation=dict(font=dict(color=CATEGORICAL[0], size=12)),
    )
    base_layout(fig, height, "Outcome per trade, in R")
    fig.update_xaxes(title="R multiple")
    fig.update_yaxes(title="Trades")
    fig.update_layout(barmode="overlay", bargap=0.05)
    return fig

def build_r_fan(path_result, realised_r=None, height: int = 380) -> go.Figure:
    """Cumulative R across resampled trade orders.

    R accumulates rather than compounds, so this is a running sum. A realised path
    hugging the top of the fan means the ordering flattered the result.
    """
    paths = path_result.paths
    x = np.arange(paths.shape[1])
    lo90, hi90 = path_result.percentile_band(5, 95)
    lo50, hi50 = path_result.percentile_band(25, 75)

    fig = go.Figure()
    for lo, hi, alpha, label in ((lo90, hi90, 0.16, "5–95%"), (lo50, hi50, 0.30, "25–75%")):
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]), y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=rgba(BAND_HUE, alpha),
            line=dict(width=0), name=label, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=x, y=path_result.median_path, mode="lines", name="Median",
        line=dict(color=BAND_HUE, width=2),
        hovertemplate="Trade %{x} · median %{y:+.1f}R<extra></extra>",
    ))

    if realised_r is not None:
        series = pd.Series(realised_r).dropna().to_numpy(dtype=float)
        if series.size:
            actual = np.concatenate([[0.0], np.cumsum(series)])
            fig.add_trace(go.Scatter(
                x=np.arange(actual.size), y=actual, mode="lines", name="Realised",
                line=dict(color=REALISED, width=2.5),
                hovertemplate="Trade %{x} · realised %{y:+.1f}R<extra></extra>",
            ))

    base_layout(fig, height, "Cumulative R across resampled trade orders")
    fig.update_xaxes(title="Trade number")
    fig.update_yaxes(title="Cumulative R")
    fig.add_hline(y=0.0, line=dict(color=INK_MUTED, width=1))
    return fig

def build_mae_vs_outcome(trades: pd.DataFrame, height: int = 380) -> go.Figure:
    """How far each trade went against you before it resolved.

    Position carries the whole message, so no colour encoding is needed. Winners
    clustered near the left edge mean the stop is sitting where trades routinely
    trade before working — tightening it would cut them off, widening it would cost
    more per loss.
    """
    mae = trades["mae_r"].astype(float)
    outcome = trades["r_multiple"].astype(float)

    fig = go.Figure(go.Scatter(
        x=mae, y=outcome, mode="markers",
        marker=dict(size=10, color=rgba(CATEGORICAL[0], 0.75),
                    line=dict(width=2, color=SURFACE)),
        text=trades["symbol"],
        hovertemplate="%{text}<br>worst excursion %{x:.2f}R<br>outcome %{y:+.2f}R<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0.0, line=dict(color=INK_MUTED, width=1))
    fig.add_vline(x=-1.0, line=dict(color=STATUS_CRITICAL, width=1.5),
                  annotation_text="stop distance", annotation_position="bottom right",
                  annotation=dict(font=dict(color=STATUS_CRITICAL, size=11)))

    base_layout(fig, height, "Worst excursion against final outcome")
    fig.update_xaxes(title="Maximum adverse excursion (R)")
    fig.update_yaxes(title="Outcome (R)")
    return fig

def build_ambiguity_bound(bound: dict, height: int = 360) -> go.Figure:
    """The range the daily-bar unknown leaves open.

    When one bar covers both the stop and the target, OHLC cannot say which came
    first. These are the two resolutions; the strategy's true result is somewhere
    between them, and a wide gap means daily bars are not enough to judge it.
    """
    pessimistic = bound["pessimistic"].equity
    optimistic = bound["optimistic"].equity
    start = float(pessimistic.iloc[0]) or 1.0

    lo = pessimistic / start
    hi = optimistic / start

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hi.index, y=hi, mode="lines", name="Target first (optimistic)",
        line=dict(color=CATEGORICAL[2], width=2),
        hovertemplate="Optimistic %{y:.2f}×<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=lo.index, y=lo, mode="lines", name="Stop first (pessimistic)",
        line=dict(color=CATEGORICAL[1], width=2), fill="tonexty",
        fillcolor=rgba(CATEGORICAL[0], 0.14),
        hovertemplate="Pessimistic %{y:.2f}×<extra></extra>",
    ))

    base_layout(fig, height, "How much the intrabar unknown is worth")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="Portfolio equity (× starting)")
    return fig
