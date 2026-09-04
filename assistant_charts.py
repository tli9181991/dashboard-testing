"""Charts for the assistant tab.

The price chart draws its support and resistance from ``strategy.merged_levels``,
so the lines on screen are the ones the entry rule is actually comparing against.
Levels detected by any other method would look authoritative and mean nothing.

Colour follows the job:

* **Moving averages** are an ordered quantity (5 → 10 → 20 → 50 sessions), so they
  take a single-hue ordinal ramp rather than four unrelated colours.
* **Support and resistance** are two identities, so they take two categorical slots
  — and every level carries a text label with its price and touch count, so the
  colour never has to carry the meaning alone.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

import strategy as strat
from viz import CATEGORICAL, INK_MUTED, INK_SECONDARY, SURFACE, base_layout, rgba

#: Ordinal blue, light → dark, one step per moving average.
RAMP_EMA = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6"]
EMA_SPANS = (5, 10, 20, 50)

RESISTANCE = CATEGORICAL[1]     # orange
SUPPORT = CATEGORICAL[2]        # aqua

#: A level touched fewer times than this is noise on the chart, not a level.
MIN_TOUCHES = 1
#: Levels further than this from the last price are off the trade's map.
MAX_DISTANCE = 0.25


def levels_for_chart(frame, price: float, max_levels: int = 8,
                     min_touches: int = MIN_TOUCHES,
                     max_distance: float = MAX_DISTANCE) -> list[tuple[float, str, int]]:
    """The levels worth drawing: near enough to matter, touched enough to be real.

    Sorted by touch count so the strongest survive the cap, because a chart with
    every level on it communicates nothing.
    """
    levels = strat.merged_levels(frame)
    keep = [
        (lv_price, kind, touches) for lv_price, kind, touches in levels
        if touches >= min_touches and price > 0
        and abs(lv_price - price) / price <= max_distance
    ]
    keep.sort(key=lambda item: (-item[2], abs(item[0] - price)))
    return keep[:max_levels]


def build_price_chart(
    symbol: str,
    df: pd.DataFrame,
    lookback: int = 180,
    params: strat.StrategyParams = strat.StrategyParams(),
    height: int = 520,
    max_levels: int = 8,
) -> go.Figure:
    """Candlestick with 5/10/20/50 EMAs and labelled support / resistance."""
    frame = strat.prepare(df, params)
    price = float(frame.df["Close"].iloc[-1])
    levels = levels_for_chart(frame, price, max_levels=max_levels)

    window = frame.df.tail(lookback)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=window.index, open=window["Open"], high=window["High"],
        low=window["Low"], close=window["Close"], name=symbol, showlegend=False,
        increasing=dict(line=dict(color=INK_SECONDARY, width=1), fillcolor=INK_SECONDARY),
        decreasing=dict(line=dict(color=INK_MUTED, width=1), fillcolor=INK_MUTED),
    ))

    close = frame.df["Close"]
    for span, colour in zip(EMA_SPANS, RAMP_EMA):
        ema = close.ewm(span=span, adjust=False).mean().tail(lookback)
        fig.add_trace(go.Scatter(
            x=ema.index, y=ema, mode="lines", name=f"{span} EMA",
            line=dict(color=colour, width=2),
            hovertemplate=f"{span} EMA %{{y:,.2f}}<extra></extra>",
        ))

    for lv_price, kind, touches in levels:
        colour = RESISTANCE if kind == "resistance" else SUPPORT
        fig.add_hline(
            y=lv_price,
            line=dict(color=colour, width=1.5, dash="dot"),
            annotation_text=f"{kind[:1].upper()} {lv_price:,.2f} ×{touches}",
            annotation_position="right",
            annotation=dict(font=dict(color=colour, size=11)),
        )

    fig.add_hline(y=price, line=dict(color=INK_SECONDARY, width=1),
                  annotation_text=f"Last {price:,.2f}",
                  annotation_position="left",
                  annotation=dict(font=dict(color=INK_SECONDARY, size=11)))

    lows = [float(window["Low"].min())] + [p for p, _, _ in levels]
    highs = [float(window["High"].max())] + [p for p, _, _ in levels]
    pad = (max(highs) - min(lows)) * 0.06 or 1.0

    base_layout(fig, height, f"{symbol} — price, moving averages and levels")
    # The level and last-price labels sit outside the plot area, so the default
    # 12px gutters clip them to a bare "R" / "S". Give them room to render.
    fig.update_layout(hovermode="x unified",
                      margin=dict(l=72, r=132, t=42, b=12))
    fig.update_xaxes(rangeslider=dict(visible=False), type="date")
    fig.update_yaxes(title="Price ($)", range=[min(lows) - pad, max(highs) + pad])
    return fig


def levels_prompt_text(symbol: str, df: pd.DataFrame,
                       params: strat.StrategyParams = strat.StrategyParams()) -> str:
    """The same levels as plain text, for the assistant's context."""
    frame = strat.prepare(df, params)
    price = float(frame.df["Close"].iloc[-1])
    levels = levels_for_chart(frame, price)
    if not levels:
        return f"{symbol}: no confirmed support or resistance near the current price."

    above = sorted([(p, t) for p, k, t in levels if k == "resistance" and p > price])
    below = sorted([(p, t) for p, k, t in levels if k == "support" and p < price],
                   reverse=True)
    parts = [f"{symbol} last {price:,.2f}."]
    if above:
        parts.append("Resistance above: " +
                     ", ".join(f"{p:,.2f} (touched {t}x)" for p, t in above[:4]))
    if below:
        parts.append("Support below: " +
                     ", ".join(f"{p:,.2f} (touched {t}x)" for p, t in below[:4]))
    return " ".join(parts)
