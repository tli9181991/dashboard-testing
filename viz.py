"""Shared chart theme.

One definition of the surface, ink, grid and palette so every figure in the app
reads as part of the same system. ``swing_charts`` and ``backtest_charts`` both
draw from here rather than each carrying its own copy.

Every palette below was checked with the data-viz validator against the dark chart
surface (#1a1a19): the categorical four pass the adjacent-pair gates, and each
ordinal ramp passes monotonicity, step separation and light-end contrast.
"""

from __future__ import annotations

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


def rgba(hex_colour: str, alpha: float) -> str:
    """Translucent fill from a palette hex — for bands behind marks, never marks."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(fig: go.Figure, height: int, title: str = "") -> go.Figure:
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
