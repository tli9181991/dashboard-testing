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
