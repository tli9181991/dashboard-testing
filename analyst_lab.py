"""Test harness for ``analyst``: scenarios, invariants and the plan chart.

Kept separate from ``analyst_app`` for the same reason ``*_charts`` are separate
from ``app``: everything here is pure and testable, and importing it does not
require Streamlit.

Three pieces, and the first is the point of the whole module:

* **Scenarios** — deterministic synthetic markets that drive every branch of the
  analyst on demand: a clean breakout, a pullback to wait for, a downtrend, a
  risk-off veto, a setup whose target is too close to pay for its stop. Testing
  an advisor against whatever the market happens to be doing today exercises one
  path and tells you nothing about the other five. Each scenario carries the
  verdict it is supposed to produce, so the harness can say *pass* or *fail*
  rather than leaving you to eyeball it.
* **Invariants** — the arithmetic that has to hold for a plan to mean anything:
  stop below the entry, targets above it, risk inside the ATR budget, size
  inside the equity cap. Shown in the app so the guarantees are visible, not
  merely claimed.
* **The chart** — the plan drawn on the price it was derived from, so a stop that
  sits in a silly place is obvious at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import plotly.graph_objects as go

import analyst
import data as data_mod
import strategy as strat
from viz import (AXIS, CATEGORICAL, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
                 RAMP_MA, STATUS_CRITICAL, STATUS_GOOD, base_layout, rgba)

BARS = 600
EMA_SPANS = (20, 50, 200)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def _uptrend(seed: int, n: int = BARS) -> pd.DataFrame:
    return data_mod.synthetic_ohlcv(n=n, seed=seed, annual_drift=0.18, annual_vol=0.28)


def _risk_on(index: pd.Index) -> pd.Series:
    """A benchmark holding above its own 200 SMA, so the regime gate opens."""
    return data_mod.synthetic_benchmark(index, seed=3, annual_drift=0.25, annual_vol=0.12)


def _risk_off(index: pd.Index) -> pd.Series:
    """The repo's default synthetic benchmark, which ends below its 200 SMA."""
    return data_mod.synthetic_benchmark(index, seed=5)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    #: What this market is, and why it is worth having in the harness.
    description: str
    prices: Callable[[], pd.DataFrame]
    benchmark: Callable[[pd.Index], pd.Series]
    #: The stance the analyst is expected to reach. None when it should error out.
    expect_stance: Optional[str] = None
    #: A fragment that must appear in one of the blockers, if any is expected.
    expect_blocker: Optional[str] = None
    expect_error: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "advancing", "Breakout in a risk-on market",
        "Price above a rising 50 and 200 EMA with the engine firing BUY. The only "
        "path that should produce a tradeable entry zone.",
        lambda: _uptrend(3), _risk_on, expect_stance="buy now",
    ),
    Scenario(
        "pullback", "Trend intact, no fresh signal",
        "An uptrend the engine is not firing on. The analyst should name the level "
        "worth waiting for — anchored below the current price — instead of endorsing "
        "a fill at whatever happens to be printing.",
        lambda: _uptrend(24), _risk_on, expect_stance="wait for the pullback",
    ),
    Scenario(
        "deep_pullback", "Pullback too deep to buy the bounce",
        "Price has fallen well under its moving averages, so the level below it is "
        "close to overhead resistance. The entry must anchor below the price and then "
        "be refused on reward:risk, never anchored above it — buying a bounce back "
        "into resistance is where this kind of plan does the most damage.",
        lambda: _uptrend(33), _risk_on,
        expect_stance="stand aside", expect_blocker="R away",
    ),
    Scenario(
        "downtrend", "Downtrend",
        "Below a falling 200 EMA. No buy price should be named at all — this is the "
        "case where a chatty advisor invents a level and calls it support.",
        lambda: data_mod.synthetic_ohlcv(n=BARS, seed=42), _risk_on,
        expect_stance="stand aside", expect_blocker="stage",
    ),
    Scenario(
        "risk_off", "Good chart, risk-off market",
        "The same advancing chart as the first scenario, under a benchmark below its "
        "200 SMA. The regime gate must veto the entry while the long-term view stays "
        "positive — the two verdicts are answering different questions.",
        lambda: _uptrend(3), _risk_off,
        expect_stance="stand aside", expect_blocker="risk-off",
    ),
    Scenario(
        "target_too_close", "Target too close to pay for the stop",
        "An advancing chart whose nearest resistance sits under 1R away. The plan "
        "must be refused on reward:risk rather than shipped with a thin target.",
        lambda: _uptrend(21), _risk_on,
        expect_stance="stand aside", expect_blocker="R away",
    ),
    Scenario(
        "short_history", "Not enough history",
        "120 bars, short of the 200 EMA. Should report that it cannot form a view "
        "rather than forming one from a warmup average.",
        lambda: data_mod.synthetic_ohlcv(n=120, seed=1), _risk_on,
        expect_error=True,
    ),
)

SCENARIOS_BY_KEY = {s.key: s for s in SCENARIOS}


def run_scenario(scenario: Scenario,
                 params: analyst.AnalystParams = analyst.AnalystParams()
                 ) -> analyst.StockAnalysis:
    """Run one scenario offline. No network, no LLM, no market."""
    prices = scenario.prices()
    return analyst.analyze(
        f"DEMO-{scenario.key.upper()}", params,
        prices=prices, benchmark_close=scenario.benchmark(prices.index),
        with_news=False, with_fundamentals=False,
    )


@dataclass(frozen=True)
class Expectation:
    ok: bool
    detail: str


def check_expectation(scenario: Scenario, analysis: analyst.StockAnalysis) -> Expectation:
    """Did the analyst reach the verdict this scenario exists to produce?"""
    if scenario.expect_error:
        return Expectation(not analysis.ok,
                           f"error reported: {analysis.error}" if not analysis.ok
                           else "expected an error, got a full analysis")
    if not analysis.ok:
        return Expectation(False, f"unexpected error: {analysis.error}")

    plan = analysis.plan
    if scenario.expect_stance and plan.stance != scenario.expect_stance:
        return Expectation(False, f"stance {plan.stance!r}, expected "
                                  f"{scenario.expect_stance!r}")
    if scenario.expect_blocker:
        if not any(scenario.expect_blocker in b for b in plan.blockers):
            return Expectation(False, f"no blocker mentioning "
                                      f"{scenario.expect_blocker!r}; got {list(plan.blockers)}")
    return Expectation(True, f"stance {plan.stance!r} as expected")


def run_all(params: analyst.AnalystParams = analyst.AnalystParams()
            ) -> list[tuple[Scenario, analyst.StockAnalysis, Expectation]]:
    """Every scenario, with its verdict and whether it matched. Used by the app."""
    out = []
    for scenario in SCENARIOS:
        analysis = run_scenario(scenario, params)
        out.append((scenario, analysis, check_expectation(scenario, analysis)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Invariants
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Invariant:
    label: str
    ok: bool
    detail: str


def plan_invariants(analysis: analyst.StockAnalysis,
                    params: analyst.AnalystParams = analyst.AnalystParams()
                    ) -> list[Invariant]:
    """The relationships that must hold for a plan to be worth acting on.

    A plan with no entry is not a failure — it has its own invariant, which is
    that a refusal states its reason.
    """
    if not analysis.ok or analysis.plan is None:
        return [Invariant("analysis produced", False, analysis.error or "no plan")]

    plan, trend = analysis.plan, analysis.trend
    if plan.entry_low is None:
        return [
            Invariant("no entry names no price", plan.stop is None and plan.target1 is None,
                      "entry, stop and targets are all absent"),
            Invariant("refusal states a reason", bool(plan.blockers),
                      "; ".join(plan.blockers) or "no reason given"),
        ]

    entry_mid = (plan.entry_low + plan.entry_high) / 2
    budget = params.max_risk_atr * trend.atr
    cap = params.equity * params.max_position_pct

    return [
        Invariant("entry zone ordered", plan.entry_low <= plan.entry_high,
                  f"{plan.entry_low:,.2f} – {plan.entry_high:,.2f}"),
        Invariant("stop below entry", plan.stop < plan.entry_low,
                  f"stop {plan.stop:,.2f} vs entry low {plan.entry_low:,.2f}"),
        Invariant("targets above entry and ordered",
                  plan.entry_high < plan.target1 < plan.target2,
                  f"{plan.target1:,.2f} then {plan.target2:,.2f}"),
        Invariant("risk inside the ATR budget", entry_mid - plan.stop <= budget + 1e-6,
                  f"{entry_mid - plan.stop:,.2f} risked vs "
                  f"{params.max_risk_atr:g} ATR = {budget:,.2f}"),
        Invariant("reward:risk meets the minimum",
                  plan.reward_risk_1 >= params.min_reward_risk,
                  f"{plan.reward_risk_1:.2f}R (minimum {params.min_reward_risk:g}R)"
                  + ("" if "confirmed" in plan.target1_source
                     else " — projected target, so this is chosen, not verified")),
        Invariant("position inside the equity cap", plan.notional <= cap + 1e-6,
                  f"${plan.notional:,.0f} vs cap ${cap:,.0f}"),
        Invariant("dollar risk matches size × stop distance",
                  abs(plan.dollar_risk - plan.quantity * plan.risk_per_share) < 1e-6,
                  f"${plan.dollar_risk:,.0f}"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. The plan, drawn on the price it came from
# ─────────────────────────────────────────────────────────────────────────────

def build_plan_chart(analysis: analyst.StockAnalysis, df: pd.DataFrame,
                     lookback: int = 180, height: int = 560) -> go.Figure:
    """Candlestick with the moving averages, the entry zone, the stop and the targets.

    Status colours are used here for exactly what they mean — the stop is the loss
    and the targets are the gain — which is the one place ``viz`` reserves them for.
    """
    prepared = strat.prepare(df, strat.StrategyParams(ema_spans=analyst.EMA_SPANS))
    window = prepared.df.tail(lookback)
    symbol = analysis.ticker

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=window.index, open=window["Open"], high=window["High"],
        low=window["Low"], close=window["Close"], name=symbol, showlegend=False,
        increasing=dict(line=dict(color=INK_SECONDARY, width=1), fillcolor=INK_SECONDARY),
        decreasing=dict(line=dict(color=INK_MUTED, width=1), fillcolor=INK_MUTED),
    ))

    for span, colour in zip(EMA_SPANS, RAMP_MA):
        ema = prepared.df[f"EMA_{span}"].tail(lookback)
        fig.add_trace(go.Scatter(
            x=ema.index, y=ema, mode="lines", name=f"{span} EMA",
            line=dict(color=colour, width=2),
            hovertemplate=f"{span} EMA %{{y:,.2f}}<extra></extra>",
        ))

    marks: list[float] = [float(window["Low"].min()), float(window["High"].max())]
    plan = analysis.plan

    if plan and plan.entry_low is not None:
        fig.add_hrect(
            y0=plan.entry_low, y1=plan.entry_high,
            fillcolor=rgba(CATEGORICAL[0], 0.22), line_width=0,
            annotation_text=f"Buy {plan.entry_low:,.2f}–{plan.entry_high:,.2f}",
            annotation_position="top left",
            annotation=dict(font=dict(color=CATEGORICAL[0], size=11)),
        )
        for value, label, colour, dash in (
            (plan.stop, f"Stop {plan.stop:,.2f}", STATUS_CRITICAL, "solid"),
            (plan.target1, f"T1 {plan.target1:,.2f} ({plan.reward_risk_1:.1f}R)",
             STATUS_GOOD, "dash"),
            (plan.target2, f"T2 {plan.target2:,.2f} ({plan.reward_risk_2:.1f}R)",
             STATUS_GOOD, "dot"),
        ):
            fig.add_hline(y=value, line=dict(color=colour, width=1.5, dash=dash),
                          annotation_text=label, annotation_position="right",
                          annotation=dict(font=dict(color=colour, size=11)))
            marks.append(value)
        marks.extend([plan.entry_low, plan.entry_high])

    if analysis.trend:
        price = analysis.trend.price
        fig.add_hline(y=price, line=dict(color=INK_PRIMARY, width=1),
                      annotation_text=f"Last {price:,.2f}",
                      annotation_position="left",
                      annotation=dict(font=dict(color=INK_PRIMARY, size=11)))
        marks.append(price)

    pad = (max(marks) - min(marks)) * 0.06 or 1.0
    title = f"{symbol} — {plan.stance}" if plan else symbol
    base_layout(fig, height, title)
    fig.update_layout(hovermode="x unified", margin=dict(l=72, r=150, t=42, b=12))
    fig.update_xaxes(rangeslider=dict(visible=False), type="date")
    fig.update_yaxes(title="Price ($)", range=[min(marks) - pad, max(marks) + pad])
    return fig
