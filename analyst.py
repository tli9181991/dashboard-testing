"""Whole-position analysis for a single symbol.

Answers the questions actually asked of a position — where to buy, where the stop
goes, what the targets are, and whether to keep holding it — by combining the
pieces the app already has: ``strategy`` for the signal and the levels,
``regime`` for the market gate, ``sizing`` for the size, ``fundamentals`` for the
business and ``sentiment`` for the news.

The design rule, and the reason this module exists at all
---------------------------------------------------------
**Every number here is computed. The model only narrates.**

A language model asked "what's a good entry for NVDA" will produce a number, and
that number will look exactly as confident as one derived from an ATR and a
confirmed support level. It is the single most dangerous failure mode in an
LLM trading assistant, because the output is indistinguishable from analysis.

So the split is absolute: ``analyze()`` returns a fully-computed ``StockAnalysis``
with every price already decided by tested code, and ``narrate()`` hands that
object to the model with instructions to explain it and invent nothing. If the
LLM is unreachable the analysis is unaffected — you lose the prose, not the plan.

Causality
---------
Everything derives from ``strategy.prepare`` over completed bars and from
backward-looking rolling windows, so a plan produced for today never depends on a
bar that had not printed. ``tests/test_analyst.py`` asserts this the same way
``tests/test_causality.py`` does for the signal: appending future bars must not
change a plan already given.

Not advice. Every threshold in ``AnalystParams`` is a starting value.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import data as data_mod
import regime as regime_mod
import sizing as sizing_mod
import strategy as strat

#: EMA set the trend read needs. 50 is not in the strategy's own defaults.
EMA_SPANS = (5, 10, 20, 50, 200)
TRADING_DAYS = 252


@dataclass(frozen=True)
class AnalystParams:
    """Knobs. Mirrors the swing screener's risk constants so the two agree."""

    equity: float = 100_000.0
    target_vol: float = 0.15
    max_position_pct: float = 0.25
    #: Stop sits this many ATR below the support it hangs off.
    stop_buffer_atr: float = 0.25
    #: A stop further than this from entry is unsizeable, not conservative.
    max_risk_atr: float = 2.75
    #: Below this reward:risk the trade is reported as not worth taking.
    min_reward_risk: float = 2.0
    #: Fallback targets when no confirmed resistance sits overhead. Kept distinct
    #: from ``min_reward_risk``: projecting a target at exactly the minimum and
    #: then testing it against that minimum is a check that cannot fail.
    tp1_r_multiple: float = 2.0
    tp2_r_multiple: float = 2.5
    #: Half-width of an entry zone anchored on a level, in ATR.
    entry_band_atr: float = 0.25
    #: Lookback for relative strength against the benchmark.
    rs_lookback: int = 126
    #: Lookback for the long-term relative strength test.
    rs_long_lookback: int = 252
    #: Slope of the 200 EMA below this magnitude reads as flat, not trending.
    flat_slope: float = 0.02
    slope_window: int = 20
    benchmark: str = "^GSPC"
    period: str = "3y"
    news_days: int = 2


@dataclass(frozen=True)
class TrendState:
    as_of: str
    price: float
    ema20: float
    ema50: float
    ema200: float
    ema200_slope: float
    sma_exit: float
    atr: float
    atr_pct: float
    high_52w: float
    low_52w: float
    pct_from_high: float
    pct_above_low: float
    rs_vs_benchmark: Optional[float]
    regime_ok: Optional[bool]
    stage: str
    stage_detail: str


@dataclass(frozen=True)
class TradePlan:
    #: What the breakout engine says on the last closed bar.
    action: str
    #: What to actually do about it, in words: buy now, wait, or stand aside.
    stance: str
    entry_low: Optional[float]
    entry_high: Optional[float]
    stop: Optional[float]
    target1: Optional[float]
    target2: Optional[float]
    target1_source: str
    target2_source: str
    risk_per_share: Optional[float]
    reward_risk_1: Optional[float]
    reward_risk_2: Optional[float]
    quantity: float
    notional: float
    dollar_risk: float
    risk_pct_of_equity: float
    #: The strategy's own trailing exit — where a held position is sold on strength fading.
    trailing_exit: float
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        return self.entry_low is not None and not self.blockers


@dataclass(frozen=True)
class Criterion:
    label: str
    state: str  # "pass" | "fail" | "unknown"
    detail: str


#: Criteria derivable from price alone. A verdict resting only on these is a
#: technical read on the chart, not a view on the business, and must say so —
#: "keep holding for the long term" means something different when nothing about
#: the company was available to check.
PRICE_CRITERIA = frozenset({
    "Above the 200 EMA",
    "200 EMA rising",
    "Outperforming the benchmark (12m)",
    "Within 25% of the 52-week high",
})


@dataclass(frozen=True)
class LongTermView:
    verdict: str
    passed: int
    known: int
    criteria: tuple[Criterion, ...]
    #: "price and business" when any fundamental or news input was available,
    #: "price only" when the verdict rests entirely on the chart.
    basis: str = "price only"

    @property
    def unknowns(self) -> tuple[str, ...]:
        return tuple(c.label for c in self.criteria if c.state == "unknown")

    @property
    def price_only(self) -> bool:
        return self.basis == "price only"


@dataclass(frozen=True)
class StockAnalysis:
    ticker: str
    error: str = ""
    trend: Optional[TrendState] = None
    plan: Optional[TradePlan] = None
    long_term: Optional[LongTermView] = None
    support: tuple[dict, ...] = ()
    resistance: tuple[dict, ...] = ()
    news_text: str = ""
    fundamentals_text: str = ""
    signal_logs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error and self.trend is not None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "error": self.error,
            "trend": asdict(self.trend) if self.trend else None,
            "plan": asdict(self.plan) if self.plan else None,
            "long_term": {
                "verdict": self.long_term.verdict,
                "passed": self.long_term.passed,
                "known": self.long_term.known,
                "basis": self.long_term.basis,
                "criteria": [asdict(c) for c in self.long_term.criteria],
            } if self.long_term else None,
            "support": list(self.support),
            "resistance": list(self.resistance),
            "news": self.news_text,
            "fundamentals": self.fundamentals_text,
            "signal_logs": list(self.signal_logs),
        }

    def render(self) -> str:
        return render_analysis(self)


# ─────────────────────────────────────────────────────────────────────────────
# Trend
# ─────────────────────────────────────────────────────────────────────────────

def _aligned_benchmark(bench_close: pd.Series, index: pd.Index) -> pd.Series:
    """Benchmark closes on the symbol's calendar. Forward fill only looks back."""
    series = pd.Series(bench_close).astype(float).dropna()
    if series.empty:
        return pd.Series(dtype=float, index=index)
    combined = series.reindex(series.index.union(index)).ffill()
    return combined.reindex(index)


def _return_over(series: pd.Series, lookback: int) -> Optional[float]:
    clean = series.dropna()
    if len(clean) <= lookback:
        return None
    start = float(clean.iloc[-lookback - 1])
    if start <= 0:
        return None
    return float(clean.iloc[-1]) / start - 1.0


def _relative_strength(close: pd.Series, bench: pd.Series, lookback: int) -> Optional[float]:
    own = _return_over(close, lookback)
    theirs = _return_over(bench, lookback) if not bench.empty else None
    if own is None or theirs is None:
        return None
    return own - theirs


def classify_stage(price: float, ema20: float, ema50: float, ema200: float,
                   slope: float, params: AnalystParams) -> tuple[str, str]:
    """Weinstein-style stage read from moving averages and the 200 EMA's slope.

    Deliberately coarse. The point is to separate "pull back into strength" from
    "catch a falling knife", which is the distinction that decides whether a dip
    is an entry, and no finer classification is needed for that.
    """
    rising = slope > params.flat_slope
    falling = slope < -params.flat_slope

    if price > ema50 > ema200 and rising:
        return "advancing", ("price above a rising 50 and 200 EMA — "
                             "trend intact, dips are entries")
    if price > ema200 and ema50 > ema200 and price < ema20:
        return "pullback", ("still above the 200 EMA with the 50 above it, but "
                            "trading under the 20 — a pullback inside an uptrend")
    if price < ema200 and falling:
        return "declining", ("below a falling 200 EMA — downtrend, no long entries "
                             "until it turns")
    if price < ema200:
        return "repairing", ("below the 200 EMA but the average is not yet falling — "
                             "damaged, unproven")
    if not rising and not falling:
        return "basing", ("above a flat 200 EMA — range, no trend to follow")
    return "transition", "moving averages disagree — no clean trend read"


def trend_state(prepared: strat.StrategyFrame, decision: strat.Decision,
                bench_close: Optional[pd.Series], regime_ok: Optional[bool],
                params: AnalystParams) -> TrendState:
    df = prepared.df
    close = df["Close"]
    price = float(close.iloc[-1])
    atr = float(df["ATR"].iloc[-1])

    ema20 = float(df["EMA_20"].iloc[-1])
    ema50 = float(df["EMA_50"].iloc[-1])
    ema200 = float(df["EMA_200"].iloc[-1])

    ema200_series = df["EMA_200"].dropna()
    slope = 0.0
    if len(ema200_series) > params.slope_window:
        prior = float(ema200_series.iloc[-params.slope_window - 1])
        if prior > 0:
            slope = float(ema200_series.iloc[-1]) / prior - 1.0

    window = min(TRADING_DAYS, len(df))
    high_52w = float(df["High"].iloc[-window:].max())
    low_52w = float(df["Low"].iloc[-window:].min())

    bench = _aligned_benchmark(bench_close, df.index) if bench_close is not None else pd.Series(dtype=float)
    rs = _relative_strength(close, bench, params.rs_lookback)

    stage, detail = classify_stage(price, ema20, ema50, ema200, slope, params)
    return TrendState(
        as_of=str(df.index[-1])[:10],
        price=price, ema20=ema20, ema50=ema50, ema200=ema200,
        ema200_slope=slope, sma_exit=decision.sma_exit, atr=atr,
        atr_pct=atr / price if price > 0 else 0.0,
        high_52w=high_52w, low_52w=low_52w,
        pct_from_high=price / high_52w - 1.0 if high_52w > 0 else 0.0,
        pct_above_low=price / low_52w - 1.0 if low_52w > 0 else 0.0,
        rs_vs_benchmark=rs, regime_ok=regime_ok,
        stage=stage, stage_detail=detail,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The plan
# ─────────────────────────────────────────────────────────────────────────────

def _levels_around(prepared: strat.StrategyFrame, price: float,
                   max_levels: int = 6) -> tuple[list[dict], list[dict]]:
    merged = strat.merged_levels(prepared)
    above = sorted([(p, t) for p, k, t in merged if p > price])[:max_levels]
    below = sorted([(p, t) for p, k, t in merged if p < price], reverse=True)[:max_levels]
    to_row = lambda p, t: {"price": p, "touches": t, "distance_pct": p / price - 1.0}
    return ([to_row(p, t) for p, t in below], [to_row(p, t) for p, t in above])


def _entry_zone(trend: TrendState, decision: strat.Decision, supports: Sequence[dict],
                params: AnalystParams, strategy_params: strat.StrategyParams
                ) -> tuple[Optional[float], Optional[float], str, list[str]]:
    """Where to buy, and why there.

    Three cases, and the third is the important one: when the answer is "not
    here", the zone is None rather than a number hedged with caveats.
    """
    price, atr = trend.price, trend.atr
    band = params.entry_band_atr * atr
    notes: list[str] = []

    if decision.action == strat.Action.BUY:
        broken = decision.broken_resistance
        ceiling = price + band
        if broken:
            # The engine's own rule: an entry is only valid inside the band above
            # the level it broke. That makes a hard "do not pay more than" price.
            ceiling = min(ceiling, broken * (1 + strategy_params.breakout_band))
            notes.append(
                f"Breakout entry. The rule only holds within "
                f"{strategy_params.breakout_band:.0%} of the broken level "
                f"{broken:,.2f}, so {ceiling:,.2f} is the highest valid fill.")
        low = max(broken, price - band) if broken else price - band
        if low >= ceiling:
            low = ceiling - band
        return low, ceiling, "buy now", notes

    if trend.stage in ("advancing", "pullback"):
        # Not a fresh signal, so name the level worth waiting for rather than
        # endorsing a fill at whatever price happens to be printing.
        #
        # The anchor must sit BELOW the current price. Confirmed support and the
        # moving averages are both candidates, and the one that matters is the
        # highest of them — the first level price would meet on a further dip.
        # Anchoring above the price would propose buying a bounce back into
        # resistance and call it a pullback entry, which is where this kind of
        # plan does the most damage.
        candidates: list[tuple[float, str]] = [
            (s["price"], "confirmed support") for s in supports if s["price"] < price]
        candidates += [(ma, label) for ma, label in
                       ((trend.ema20, "the 20 EMA"), (trend.ema50, "the 50 EMA"))
                       if ma < price]
        if not candidates:
            notes.append("No confirmed support or moving average below the current "
                         "price to anchor an entry on.")
            return None, None, "stand aside", notes

        anchor, source = max(candidates, key=lambda item: item[0])
        notes.append(f"Anchored on {source} at {anchor:,.2f} — the first level price "
                     f"would meet on a further dip.")
        return anchor - band, anchor + band, "wait for the pullback", notes

    return None, None, "stand aside", notes


def _stop_for(entry_low: float, entry_mid: float, atr: float, supports: Sequence[dict],
              params: AnalystParams) -> tuple[float, str]:
    """Stop under the nearest support, but never further than the ATR budget."""
    below = [s["price"] for s in supports if s["price"] < entry_low]
    atr_floor = entry_mid - params.max_risk_atr * atr
    if below:
        structural = max(below) - params.stop_buffer_atr * atr
        if structural > atr_floor:
            return structural, (f"{params.stop_buffer_atr:g} ATR under confirmed "
                                f"support {max(below):,.2f}")
        return atr_floor, (f"support {max(below):,.2f} is more than "
                           f"{params.max_risk_atr:g} ATR away — stop capped at the "
                           f"risk budget instead")
    return atr_floor, f"no confirmed support below; {params.max_risk_atr:g} ATR under entry"


def build_plan(trend: TrendState, decision: strat.Decision, supports: Sequence[dict],
               resistances: Sequence[dict], params: AnalystParams,
               strategy_params: strat.StrategyParams,
               asset_class: strat.AssetClass) -> TradePlan:
    blockers: list[str] = []
    notes: list[str] = []

    # Conditions that rule out any new long, whatever the entry would have been.
    if trend.regime_ok is False:
        blockers.append("Market regime is risk-off; the app's own gate vetoes new entries.")
    if trend.stage in ("declining", "repairing"):
        blockers.append(f"Trend stage is '{trend.stage}' — this is not a long setup.")

    entry_low, entry_high, stance, entry_notes = _entry_zone(
        trend, decision, supports, params, strategy_params)
    notes.extend(entry_notes)

    # Price under the exit average is a reason not to buy *today*. That is what a
    # "wait for the pullback" plan already says, so treating it as a blocker there
    # would refuse the most ordinary situation there is: a good chart that has
    # dipped. It only contradicts a plan that wants filling now.
    below_exit = trend.price < trend.sma_exit
    if below_exit and stance == "buy now":
        blockers.append(
            f"Price {trend.price:,.2f} is below the {strategy_params.sma_exit} SMA "
            f"{trend.sma_exit:,.2f}, which is the strategy's own exit line.")
    elif below_exit:
        notes.append(
            f"Price {trend.price:,.2f} is under the {strategy_params.sma_exit} SMA "
            f"{trend.sma_exit:,.2f} — which is why this is a level to wait for, "
            f"not a fill to take now.")

    empty = TradePlan(
        action=decision.action.value, stance="stand aside",
        entry_low=None, entry_high=None, stop=None, target1=None, target2=None,
        target1_source="", target2_source="", risk_per_share=None,
        reward_risk_1=None, reward_risk_2=None, quantity=0.0, notional=0.0,
        dollar_risk=0.0, risk_pct_of_equity=0.0, trailing_exit=trend.sma_exit,
        blockers=tuple(blockers), notes=tuple(notes),
    )
    if entry_low is None or entry_high is None:
        return empty

    entry_mid = (entry_low + entry_high) / 2.0
    stop, stop_note = _stop_for(entry_low, entry_mid, trend.atr, supports, params)
    notes.append(f"Stop {stop_note}.")

    risk = entry_mid - stop
    if risk <= 0:
        blockers.append("Could not place a stop below the entry zone.")
        return empty

    above = [r["price"] for r in resistances if r["price"] > entry_mid]
    t1_from_level = bool(above)
    if above:
        target1, t1_src = above[0], "next confirmed resistance"
    else:
        target1 = entry_mid + params.tp1_r_multiple * risk
        t1_src = f"no resistance overhead — {params.tp1_r_multiple:g}R projection"
    if len(above) > 1:
        target2, t2_src = above[1], "second confirmed resistance"
    else:
        target2 = entry_mid + params.tp2_r_multiple * risk
        t2_src = f"{params.tp2_r_multiple:g}R projection"

    rr1 = (target1 - entry_mid) / risk
    rr2 = (target2 - entry_mid) / risk
    # The reward:risk test only carries information when the target is a level the
    # market put there. Against a projection it is circular — the number was chosen,
    # so it passes by construction — and reporting it as a check would overstate
    # what has been verified.
    if t1_from_level:
        if rr1 < params.min_reward_risk:
            blockers.append(
                f"First target is only {rr1:.2f}R away (minimum {params.min_reward_risk:g}R) — "
                f"resistance at {target1:,.2f} is too close to pay for the stop.")
    else:
        notes.append("No confirmed resistance overhead, so the first target is a "
                     "projection — its reward:risk is chosen, not verified.")

    ann_vol = sizing_mod.annualized_vol_from_atr(trend.atr, entry_mid)
    quantity = sizing_mod.target_quantity(
        params.equity, entry_mid, ann_vol, asset_class,
        sizing_mod.SizingParams(target_vol=params.target_vol,
                                max_position_pct=params.max_position_pct))
    notional = quantity * entry_mid
    dollar_risk = quantity * risk
    if quantity <= 0:
        blockers.append("Volatility targeting sizes this position at zero units.")

    return TradePlan(
        action=decision.action.value,
        stance=stance if not blockers else "stand aside",
        entry_low=entry_low, entry_high=entry_high, stop=stop,
        target1=target1, target2=target2,
        target1_source=t1_src, target2_source=t2_src,
        risk_per_share=risk, reward_risk_1=rr1, reward_risk_2=rr2,
        quantity=quantity, notional=notional, dollar_risk=dollar_risk,
        risk_pct_of_equity=dollar_risk / params.equity if params.equity else 0.0,
        trailing_exit=trend.sma_exit,
        blockers=tuple(blockers), notes=tuple(notes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# The long-term question
# ─────────────────────────────────────────────────────────────────────────────

def _fundamental(snapshot, section: str, label: str) -> Optional[float]:
    if snapshot is None:
        return None
    try:
        return snapshot.get(section, label)
    except Exception:
        return None


def long_term_view(trend: TrendState, close: pd.Series, bench: pd.Series,
                   snapshot, news_label: Optional[str],
                   params: AnalystParams) -> LongTermView:
    """A separate rubric from the trade plan, because it is a separate question.

    A name can be a bad swing entry and a fine long-term hold, or the reverse.
    Collapsing both into one BUY/SELL is where this kind of assistant usually
    goes wrong.

    Unknown inputs are counted as unknown, never as a pass or a fail. A verdict
    resting on two of eight criteria says so out loud.
    """
    criteria: list[Criterion] = []

    criteria.append(Criterion(
        "Above the 200 EMA", "pass" if trend.price > trend.ema200 else "fail",
        f"price {trend.price:,.2f} vs 200 EMA {trend.ema200:,.2f}"))
    criteria.append(Criterion(
        "200 EMA rising", "pass" if trend.ema200_slope > 0 else "fail",
        f"{trend.ema200_slope:+.2%} over {params.slope_window} sessions"))

    rs_long = _relative_strength(close, bench, params.rs_long_lookback)
    criteria.append(Criterion(
        "Outperforming the benchmark (12m)",
        "unknown" if rs_long is None else ("pass" if rs_long > 0 else "fail"),
        "not enough overlapping history" if rs_long is None else f"{rs_long:+.1%} vs benchmark"))

    criteria.append(Criterion(
        "Within 25% of the 52-week high",
        "pass" if trend.pct_from_high > -0.25 else "fail",
        f"{trend.pct_from_high:+.1%} from {trend.high_52w:,.2f}"))

    margin = _fundamental(snapshot, "Profitability", "Net margin")
    criteria.append(Criterion(
        "Profitable", "unknown" if margin is None else ("pass" if margin > 0 else "fail"),
        "net margin not reported" if margin is None else f"net margin {margin:.1%}"))

    growth = _fundamental(snapshot, "Growth", "Revenue growth (yoy)")
    criteria.append(Criterion(
        "Revenue growing", "unknown" if growth is None else ("pass" if growth > 0 else "fail"),
        "revenue growth not reported" if growth is None else f"revenue {growth:+.1%} yoy"))

    current = _fundamental(snapshot, "Balance sheet", "Current ratio")
    criteria.append(Criterion(
        "Balance sheet covers near-term liabilities",
        "unknown" if current is None else ("pass" if current >= 1.0 else "fail"),
        "current ratio not reported" if current is None else f"current ratio {current:.2f}"))

    criteria.append(Criterion(
        "News not negative",
        "unknown" if not news_label else ("fail" if news_label.lower().startswith("neg") else "pass"),
        f"latest news reads {news_label}" if news_label else "no scored news in the window"))

    known = [c for c in criteria if c.state != "unknown"]
    passed = sum(1 for c in known if c.state == "pass")
    basis = ("price and business"
             if any(c.label not in PRICE_CRITERIA for c in known) else "price only")

    if len(known) < 3:
        verdict = "INSUFFICIENT DATA"
    else:
        ratio = passed / len(known)
        if ratio >= 0.75:
            verdict = "ACCUMULATE"
        elif ratio >= 0.5:
            verdict = "HOLD"
        elif ratio >= 0.3:
            verdict = "REDUCE"
        else:
            verdict = "EXIT"

    return LongTermView(verdict, passed, len(known), tuple(criteria), basis)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _news_block(ticker: str, params: AnalystParams) -> tuple[str, Optional[str]]:
    """News text plus the sentiment label, or a plain statement that there is none.

    Imported lazily: ``sentiment`` pulls in the LangChain stack, and an analysis
    with no news is far more useful than an ImportError.
    """
    try:
        import sentiment as sentiment_mod
    except Exception as exc:
        return f"News unavailable: {type(exc).__name__}: {exc}", None

    try:
        payload = sentiment_mod.get_recent_sentiment(ticker, days=params.news_days)
        text = sentiment_mod.sentiment_prompt_text(payload)
        label = (payload.get("data") or {}).get("label")
        if payload.get("error"):
            # No LLM or no headlines. Fall back to the raw titles, which are still
            # worth reading even unscored.
            window = payload.get("window") or {}
            articles = window.get("articles") or []
            if articles:
                heads = "\n".join(
                    f"- [{a.get('publisher') or 'unknown'}] {a.get('title')}"
                    for a in articles[:6])
                text = (f"News (last {params.news_days} days, unscored — "
                        f"{payload['error']}):\n{heads}")
        return text, label
    except Exception as exc:
        return f"News unavailable: {type(exc).__name__}: {exc}", None


def _fundamentals_block(ticker: str):
    """``(snapshot, text)``. Never raises; missing fundamentals are not fatal."""
    try:
        import fundamentals as fundamentals_mod
        snapshot = fundamentals_mod.fetch(ticker)
    except Exception as exc:
        return None, f"Fundamentals unavailable: {type(exc).__name__}: {exc}"
    if snapshot is None or not getattr(snapshot, "ok", False):
        return None, f"Fundamentals unavailable: {getattr(snapshot, 'error', 'no data')}"
    try:
        return snapshot, snapshot.to_prompt_text()
    except Exception:
        return snapshot, ""


def analyze(ticker: str, params: AnalystParams = AnalystParams(), *,
            prices: Optional[pd.DataFrame] = None,
            benchmark_close: Optional[pd.Series] = None,
            with_news: bool = True, with_fundamentals: bool = True) -> StockAnalysis:
    """Full analysis of one symbol.

    ``prices`` and ``benchmark_close`` are injectable so the whole pipeline runs
    offline in tests and against the IB bars cached by
    ``notebooks/ib_tws_connect.ipynb``.
    """
    ticker = ticker.strip().upper()
    asset_class = strat.AssetClass.infer(ticker)

    frame = prices if prices is not None else data_mod.load_history(ticker, period=params.period)
    if frame is None or frame.empty:
        return StockAnalysis(ticker, error="No price history could be loaded.")

    closed = strat.drop_forming_bar(frame, asset_class)
    if closed.empty:
        closed = frame
    strategy_params = strat.StrategyParams(ema_spans=EMA_SPANS)
    if len(closed) < max(EMA_SPANS):
        return StockAnalysis(
            ticker, error=f"Only {len(closed)} bars; need at least {max(EMA_SPANS)} "
                          f"for the 200 EMA trend read.")

    if benchmark_close is None and prices is None:
        bench_frame = data_mod.load_history(params.benchmark, period=params.period)
        benchmark_close = bench_frame["Close"] if not bench_frame.empty else None

    regime_ok: Optional[bool] = None
    if benchmark_close is not None and len(benchmark_close):
        gate = regime_mod.build_gate(benchmark_close, closed.index)
        regime_ok = bool(gate.iloc[-1])

    prepared, decision = strat.evaluate_latest(
        closed, strat.Position(), strategy_params,
        True if regime_ok is None else regime_ok)

    trend = trend_state(prepared, decision, benchmark_close, regime_ok, params)
    supports, resistances = _levels_around(prepared, trend.price)
    plan = build_plan(trend, decision, supports, resistances, params,
                      strategy_params, asset_class)

    news_text, news_label = _news_block(ticker, params) if with_news else ("", None)
    snapshot, fundamentals_text = _fundamentals_block(ticker) if with_fundamentals else (None, "")

    bench = (_aligned_benchmark(benchmark_close, prepared.index)
             if benchmark_close is not None else pd.Series(dtype=float))
    long_term = long_term_view(trend, prepared.df["Close"], bench, snapshot,
                               news_label, params)

    return StockAnalysis(
        ticker=ticker, trend=trend, plan=plan, long_term=long_term,
        support=tuple(supports), resistance=tuple(resistances),
        news_text=news_text, fundamentals_text=fundamentals_text,
        signal_logs=tuple(decision.logs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _money(value: Optional[float]) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:,.2f}"


def render_analysis(analysis: StockAnalysis) -> str:
    """Plain text. This is what the model reads and what the CLI prints."""
    if not analysis.ok:
        return f"{analysis.ticker}: {analysis.error}"

    t, plan, lt = analysis.trend, analysis.plan, analysis.long_term
    lines = [
        f"=== {analysis.ticker} — analysis as of {t.as_of} (last CLOSED bar) ===",
        "",
        "CURRENT STATE",
        f"  Price            {_money(t.price)}",
        f"  Trend stage      {t.stage} — {t.stage_detail}",
        f"  Moving averages  20 EMA {_money(t.ema20)} | 50 EMA {_money(t.ema50)} | "
        f"200 EMA {_money(t.ema200)} ({t.ema200_slope:+.2%} over 20 sessions)",
        f"  Volatility       ATR {_money(t.atr)} ({t.atr_pct:.1%} of price)",
        f"  52-week range    {_money(t.low_52w)} – {_money(t.high_52w)} "
        f"({t.pct_from_high:+.1%} from the high, {t.pct_above_low:+.1%} off the low)",
    ]
    if t.rs_vs_benchmark is not None:
        lines.append(f"  Relative strength {t.rs_vs_benchmark:+.1%} vs benchmark over 6 months")
    if t.regime_ok is not None:
        lines.append(f"  Market regime    {'risk-on' if t.regime_ok else 'RISK-OFF (new entries vetoed)'}")
    lines.append(f"  Engine signal    {plan.action}")

    lines += ["", "LEVELS"]
    for key, label in (("resistance", "  Resistance above"), ("support", "  Support below")):
        rows = getattr(analysis, key)
        if rows:
            lines.append(f"{label}: " + ", ".join(
                f"{r['price']:,.2f} ({r['distance_pct']:+.1%}, {r['touches']}x)" for r in rows[:4]))
    if not analysis.support and not analysis.resistance:
        lines.append("  No confirmed levels near the current price.")

    lines += ["", f"TRADE PLAN — {plan.stance.upper()}"]
    if plan.blockers:
        for blocker in plan.blockers:
            lines.append(f"  ✗ {blocker}")
    if plan.entry_low is not None:
        if plan.blockers:
            # The levels below are still correct arithmetic, but a reader who takes
            # the zone and skips the refusal above has done the one thing this
            # report exists to prevent. Say so between the two.
            lines.append("  The levels below are REFERENCE ONLY — the blockers above "
                         "mean this is not a trade today.")
        lines += [
            f"  Buy zone         {_money(plan.entry_low)} – {_money(plan.entry_high)}",
            f"  Stop (sell)      {_money(plan.stop)}  "
            f"(risk {_money(plan.risk_per_share)}/share)",
            f"  Target 1         {_money(plan.target1)}  "
            f"({plan.reward_risk_1:.2f}R — {plan.target1_source})",
            f"  Target 2         {_money(plan.target2)}  "
            f"({plan.reward_risk_2:.2f}R — {plan.target2_source})",
            f"  Trailing exit    {_money(plan.trailing_exit)} "
            f"(the strategy sells a close below this)",
            f"  Size             {plan.quantity:,.0f} units = ${plan.notional:,.0f}, "
            f"${plan.dollar_risk:,.0f} at risk ({plan.risk_pct_of_equity:.2%} of equity)",
        ]
    else:
        lines.append("  No entry. There is no buy price worth naming here.")
    for note in plan.notes:
        lines.append(f"  · {note}")

    lines += ["", f"LONG-TERM HOLD — {lt.verdict}",
              f"  {lt.passed} of {lt.known} known criteria pass"
              + (f"; {len(lt.unknowns)} unknown" if lt.unknowns else "")
              + f" (basis: {lt.basis})"]
    if lt.price_only:
        lines.append("  ! Nothing about the business was available — this is a read "
                     "on the chart, not on the company.")
    mark = {"pass": "✓", "fail": "✗", "unknown": "?"}
    for c in lt.criteria:
        lines.append(f"  {mark[c.state]} {c.label}: {c.detail}")

    if analysis.news_text:
        lines += ["", "NEWS", *(f"  {line}" for line in analysis.news_text.splitlines())]
    if analysis.fundamentals_text:
        lines += ["", "FUNDAMENTALS",
                  *(f"  {line}" for line in analysis.fundamentals_text.splitlines()[:20])]

    lines += ["", "Computed by the app's own tested engines. Not advice; every "
                  "threshold is a starting value, not a validated one."]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Narration — the only place a model touches any of this
# ─────────────────────────────────────────────────────────────────────────────

NARRATOR_SYSTEM = (
    "You are a trading analyst writing up an analysis that has ALREADY been computed.\n\n"
    "Absolute rule: every number in your answer must be copied from the report below. "
    "Do not calculate, estimate, adjust or invent any price, level, target, ratio or "
    "position size. If the report does not contain a number, say it is not available. "
    "You may not derive new levels from the ones given.\n\n"
    "Write for someone deciding what to do with this position today. Cover, in order: "
    "what the stock is doing now; whether to buy and in what zone; where the stop and "
    "targets sit; and whether to keep holding it long term. Lead with anything in the "
    "'blockers' list — a refusal is a useful answer, and a plan with blockers must be "
    "reported as no-trade rather than softened into a maybe. When the report marks its levels REFERENCE ONLY, you must not present them as an entry to take. State the long-term verdict "
    "with how many criteria are unknown, so the reader knows how much it rests on.\n\n"
    "Be direct and brief. No disclaimers beyond the one already in the report."
)


def narrate(analysis: StockAnalysis, question: str = "") -> str:
    """Prose over a computed analysis. Returns a plain message if no LLM is configured.

    The model gets the finished report and a rule against arithmetic. It is a
    writer here, not an analyst — which is the only role an LLM can safely hold
    in a path that outputs prices.
    """
    if not analysis.ok:
        return f"{analysis.ticker}: {analysis.error}"

    try:
        from config import (AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_CREDENTIAL,
                            DEEPSEEK_MODEL_NAME)
        if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_CREDENTIAL:
            return analysis.render()
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
    except Exception:
        return analysis.render()

    prompt = f"<report>\n{analysis.render()}\n</report>"
    if question:
        prompt += f"\n\nThe user asked: {question}"

    try:
        llm = AzureAIChatCompletionsModel(
            endpoint=AZURE_INFERENCE_ENDPOINT,
            credential=AZURE_INFERENCE_CREDENTIAL,
            model=DEEPSEEK_MODEL_NAME,
            temperature=0.1,
        )
        return str(llm.invoke([("system", NARRATOR_SYSTEM), ("user", prompt)]).content)
    except Exception as exc:
        return f"{analysis.render()}\n\n(Narration unavailable: {exc})"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse one symbol end to end.")
    parser.add_argument("ticker")
    parser.add_argument("--equity", type=float, default=AnalystParams.equity)
    parser.add_argument("--target-vol", type=float, default=AnalystParams.target_vol)
    parser.add_argument("--period", default=AnalystParams.period)
    parser.add_argument("--news-days", type=int, default=AnalystParams.news_days)
    parser.add_argument("--no-news", action="store_true", help="skip the news lookup")
    parser.add_argument("--no-fundamentals", action="store_true")
    parser.add_argument("--narrate", action="store_true",
                        help="have the LLM write it up (needs Azure credentials)")
    parser.add_argument("--json", action="store_true", help="emit the raw analysis as JSON")
    args = parser.parse_args(argv)

    params = AnalystParams(equity=args.equity, target_vol=args.target_vol,
                           period=args.period, news_days=args.news_days)
    analysis = analyze(args.ticker, params, with_news=not args.no_news,
                       with_fundamentals=not args.no_fundamentals)

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2, default=str))
    elif args.narrate:
        print(narrate(analysis))
    else:
        print(analysis.render())
    return 0 if analysis.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
