"""Callable analysis tools for the assistant agent.

Every function here is pure Python returning a dict, with a matching ``render_*``
that turns it into the plain text the model reads. The LangChain ``@tool``
wrappers in ``chat_agent`` are glue over these — which is what makes them
testable without a model, a network or an API key.

The point of these tools is not to give the model more to read. It is to let it
*compute*. A number that comes back from ``strategy.evaluate`` or
``simulation.random_entry_benchmark`` carries the tests behind those modules; a
number the model paraphrases from a context blob carries nothing. So each tool
wraps machinery the app already has rather than reimplementing it, and every one
of them can come back saying it does not know.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

import pandas as pd

import data as data_mod
import regime as regime_mod
import simulation as sim
import sizing as sizing_mod
import strategy as strat
import swing_screener as swing
from backtest import BacktestConfig, run_backtest

DEFAULT_PERIOD = "3y"
#: Trading sessions before / after a report that a swing trade should avoid.
#: Mirrors swing_screener's §05 defaults so both layers agree.
BLACKOUT_BEFORE = swing.CFG["earnings_blackout_before"]
BLACKOUT_AFTER = swing.CFG["earnings_blackout_after"]
#: Extra calendar days of caution when the vendor's date is an estimate.
ESTIMATE_BUFFER = swing.CFG["earnings_date_buffer"]


def _prices(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    return data_mod.load_history(ticker.strip().upper(), period=period)


def _completed(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop a still-forming final bar, as every other path in the app does."""
    closed = strat.drop_forming_bar(df, strat.AssetClass.infer(ticker))
    return closed if not closed.empty else df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Earnings calendar
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_dates(value: Any) -> list[date]:
    out: list[date] = []
    for item in (value if isinstance(value, (list, tuple)) else [value]):
        if item is None:
            continue
        if isinstance(item, datetime):
            out.append(item.date())
        elif isinstance(item, date):
            out.append(item)
        else:
            try:
                out.append(pd.Timestamp(item).date())
            except Exception:
                continue
    return sorted(set(out))


def earnings_calendar(ticker: str, asof: Optional[date] = None) -> dict:
    """Next scheduled report, and whether a swing entry today sits in its blackout.

    yfinance has shipped several shapes for this over the years — a dict, a frame,
    and a separate ``earnings_dates`` table — so all three are tried. A date range
    rather than a single date means the vendor is estimating, which widens the
    window rather than narrowing it: vendors are routinely a week out, and this is
    the most common way a well-selected swing trade dies.
    """
    ticker = ticker.strip().upper()
    asof = asof or datetime.now(timezone.utc).date()
    result: dict = {"ticker": ticker, "asof": asof.isoformat(), "next_date": None,
                    "days_away": None, "confirmed": None, "in_blackout": False,
                    "reason": "", "error": ""}

    try:
        import yfinance as yf
    except ImportError:
        result["error"] = "yfinance is not installed"
        return result

    dates: list[date] = []
    try:
        handle = yf.Ticker(ticker)
        calendar = getattr(handle, "calendar", None)
        if isinstance(calendar, dict):
            dates = _coerce_dates(calendar.get("Earnings Date"))
        elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
            if "Earnings Date" in calendar.index:
                dates = _coerce_dates(list(calendar.loc["Earnings Date"]))
            elif "Earnings Date" in calendar.columns:
                dates = _coerce_dates(list(calendar["Earnings Date"]))
        if not dates:
            table = getattr(handle, "earnings_dates", None)
            if isinstance(table, pd.DataFrame) and not table.empty:
                dates = _coerce_dates(list(table.index))
    except Exception as exc:
        result["error"] = f"Earnings lookup failed: {exc}"
        return result

    upcoming = [d for d in dates if d >= asof]
    if not upcoming:
        result["reason"] = "No future earnings date published for this symbol."
        return result

    # Two or more clustered dates is a vendor range, i.e. an estimate.
    confirmed = len([d for d in dates if d >= asof]) == 1
    next_date = upcoming[0]
    days = (next_date - asof).days

    # Sessions to calendar days, plus a buffer when the date is a guess.
    buffer_days = 0 if confirmed else ESTIMATE_BUFFER
    before = BLACKOUT_BEFORE * 7 / 5 + buffer_days
    after = BLACKOUT_AFTER * 7 / 5 + buffer_days
    recent = [d for d in dates if d < asof]
    since_last = (asof - recent[-1]).days if recent else None

    in_blackout = days <= before or (since_last is not None and since_last <= after)

    result.update({
        "next_date": next_date.isoformat(), "days_away": days,
        "confirmed": confirmed, "in_blackout": bool(in_blackout),
        "blackout_before_days": round(before, 1),
        "blackout_after_days": round(after, 1),
        "days_since_last": since_last,
        "reason": (f"Reports in {days} day(s) — inside the "
                   f"{before:.0f}-day pre-earnings blackout." if in_blackout and days <= before
                   else f"Reported {since_last} day(s) ago — inside the post-earnings blackout."
                   if in_blackout else f"Next report in {days} day(s), outside the blackout."),
    })
    return result


def earnings_map(tickers, asof: Optional[date] = None) -> dict[str, tuple[date, bool]]:
    """``{ticker: (date, confirmed)}`` in the shape ``swing_screener.earnings_veto``
    expects, so §05 stops being a no-op."""
    out: dict[str, tuple[date, bool]] = {}
    for ticker in tickers:
        info = earnings_calendar(ticker, asof=asof)
        if info.get("next_date"):
            out[info["ticker"]] = (date.fromisoformat(info["next_date"]),
                                   bool(info["confirmed"]))
    return out


def render_earnings(info: dict) -> str:
    if info["error"]:
        return f"Earnings for {info['ticker']}: {info['error']}"
    if not info["next_date"]:
        return f"Earnings for {info['ticker']}: {info['reason']}"
    certainty = "confirmed" if info["confirmed"] else "ESTIMATED (treat as approximate)"
    flag = "BLACKOUT — do not open a swing position" if info["in_blackout"] else "clear"
    return (f"Earnings for {info['ticker']}: {info['next_date']} "
            f"({info['days_away']} days away, {certainty}). Status: {flag}. "
            f"{info['reason']}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trade plan validation — the refusal tool
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def validate_trade_plan(ticker: str, entry: float, stop: float,
                        target: Optional[float] = None,
                        equity: float = 100_000.0,
                        max_position_pct: float = 0.25,
                        min_reward_risk: float = 2.0,
                        period: str = DEFAULT_PERIOD) -> dict:
    """Check a proposed long against the account's own risk rules.

    Returns a verdict plus every check, so a refusal can say which rule it broke.
    This is the tool that lets the assistant say no: a model asked to respect a
    risk limit in a system prompt will drift, one handed a FAIL with a reason
    has something concrete to report.
    """
    ticker = ticker.strip().upper()
    checks: list[Check] = []
    entry, stop = float(entry), float(stop)

    risk_per_share = entry - stop
    checks.append(Check(
        "stop below entry", risk_per_share > 0,
        f"entry {entry:,.2f}, stop {stop:,.2f}, risk {risk_per_share:,.2f}/share"
        if risk_per_share > 0 else
        f"stop {stop:,.2f} is not below entry {entry:,.2f} — this is not a long",
    ))

    frame = _prices(ticker, period)
    atr = None
    if frame.empty:
        checks.append(Check("price history", False, "no price history could be loaded"))
    else:
        closed = _completed(frame, ticker)
        prepared = strat.prepare(closed)
        atr = float(prepared.df["ATR"].iloc[-1])
        last = float(prepared.df["Close"].iloc[-1])
        checks.append(Check("price history", True,
                            f"{len(closed)} bars, last {last:,.2f}, ATR {atr:,.2f}"))

        if risk_per_share > 0 and atr > 0:
            in_atr = risk_per_share / atr
            cap = swing.CFG["max_risk_atr"]
            checks.append(Check(
                "stop within ATR budget", in_atr <= cap,
                f"stop is {in_atr:.2f} ATR from entry (limit {cap})",
            ))

    if target is not None and risk_per_share > 0:
        reward = float(target) - entry
        rr = reward / risk_per_share
        checks.append(Check(
            "reward:risk", rr >= min_reward_risk,
            f"{rr:.2f}R to target {float(target):,.2f} (minimum {min_reward_risk}R)",
        ))

    shares, notional = 0, 0.0
    if risk_per_share > 0 and atr and atr > 0 and entry > 0:
        ann_vol = sizing_mod.annualized_vol_from_atr(atr, entry)
        params = sizing_mod.SizingParams(max_position_pct=max_position_pct)
        shares = sizing_mod.target_quantity(
            equity, entry, ann_vol, strat.AssetClass.infer(ticker), params)
        notional = shares * entry
        checks.append(Check(
            "position cap", notional <= equity * max_position_pct + 1e-6,
            f"vol-targeted size {shares:,.0f} units = ${notional:,.0f} "
            f"({notional / equity:.1%} of ${equity:,.0f}, cap {max_position_pct:.0%})",
        ))
        checks.append(Check(
            "size is tradeable", shares >= 1,
            f"{shares:,.0f} units at this volatility and equity",
        ))

    earnings = earnings_calendar(ticker)
    if earnings["error"]:
        checks.append(Check("earnings blackout", True,
                            f"unknown — {earnings['error']}"))
    else:
        checks.append(Check("earnings blackout", not earnings["in_blackout"],
                            earnings["reason"] or "no earnings date published"))

    failures = [c for c in checks if not c.passed]
    return {
        "ticker": ticker, "entry": entry, "stop": stop, "target": target,
        "risk_per_share": risk_per_share, "shares": shares, "notional": notional,
        "dollar_risk": shares * risk_per_share if risk_per_share > 0 else 0.0,
        "verdict": "PASS" if not failures else "FAIL",
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "failed": [c.name for c in failures],
    }


def render_validation(result: dict) -> str:
    lines = [f"Trade plan for {result['ticker']}: {result['verdict']}"]
    if result["verdict"] == "FAIL":
        lines.append("Failed: " + ", ".join(result["failed"]))
    for check in result["checks"]:
        lines.append(f"  [{'ok ' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    if result["shares"]:
        lines.append(f"  Size {result['shares']:,.0f} units, "
                     f"${result['dollar_risk']:,.0f} at risk to the stop.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal, levels, size
# ─────────────────────────────────────────────────────────────────────────────

def check_signal_now(ticker: str, period: str = DEFAULT_PERIOD,
                     use_regime: bool = True) -> dict:
    """What the breakout engine says about this symbol on the last closed bar."""
    ticker = ticker.strip().upper()
    frame = _prices(ticker, period)
    if frame.empty:
        return {"ticker": ticker, "error": "No price history could be loaded."}

    closed = _completed(frame, ticker)
    regime_ok, regime_note = True, "regime gate not applied"
    if use_regime:
        bench = _prices("^GSPC", period)
        if not bench.empty:
            gate = regime_mod.build_gate(bench["Close"], closed.index)
            regime_ok = bool(gate.iloc[-1])
            regime_note = "risk-on" if regime_ok else "risk-off (new entries vetoed)"

    prepared, decision = strat.evaluate_latest(
        closed, strat.Position(), strat.StrategyParams(), regime_ok)
    return {
        "ticker": ticker, "error": "",
        "as_of": str(prepared.index[-1])[:10],
        "action": decision.action.value,
        "price": decision.price, "sma_exit": decision.sma_exit, "atr": decision.atr,
        "broken_resistance": decision.broken_resistance,
        "next_resistance": decision.next_resistance,
        "regime_ok": regime_ok, "regime_note": regime_note,
        "logs": list(decision.logs),
    }


def render_signal(info: dict) -> str:
    if info.get("error"):
        return f"Signal for {info['ticker']}: {info['error']}"
    parts = [
        f"Breakout engine on {info['ticker']} as of {info['as_of']} (last CLOSED bar): "
        f"{info['action']}.",
        f"Price {info['price']:,.2f}, 10 SMA {info['sma_exit']:,.2f}, ATR {info['atr']:,.2f}.",
        f"Market regime: {info['regime_note']}.",
    ]
    if info.get("next_resistance"):
        parts.append(f"Next resistance {info['next_resistance']:,.2f}.")
    parts.extend(info["logs"])
    return " ".join(parts)


def support_resistance(ticker: str, period: str = DEFAULT_PERIOD,
                       max_levels: int = 6) -> dict:
    """Confirmed levels the entry rule compares against, split around the price."""
    ticker = ticker.strip().upper()
    frame = _prices(ticker, period)
    if frame.empty:
        return {"ticker": ticker, "error": "No price history could be loaded."}

    prepared = strat.prepare(_completed(frame, ticker))
    price = float(prepared.df["Close"].iloc[-1])
    levels = strat.merged_levels(prepared)

    above = sorted([(p, t) for p, k, t in levels if p > price])[:max_levels]
    below = sorted([(p, t) for p, k, t in levels if p < price], reverse=True)[:max_levels]
    return {
        "ticker": ticker, "error": "", "price": price,
        "resistance": [{"price": p, "touches": t,
                        "distance_pct": (p / price - 1)} for p, t in above],
        "support": [{"price": p, "touches": t,
                     "distance_pct": (p / price - 1)} for p, t in below],
    }


def render_levels(info: dict) -> str:
    if info.get("error"):
        return f"Levels for {info['ticker']}: {info['error']}"
    lines = [f"{info['ticker']} last {info['price']:,.2f}."]
    for key, label in (("resistance", "Resistance above"), ("support", "Support below")):
        rows = info[key]
        if rows:
            lines.append(f"{label}: " + ", ".join(
                f"{r['price']:,.2f} ({r['distance_pct']:+.1%}, touched {r['touches']}x)"
                for r in rows))
    if not info["resistance"] and not info["support"]:
        lines.append("No confirmed levels near the current price.")
    return " ".join(lines)


def size_position(ticker: str, equity: float = 100_000.0, target_vol: float = 0.15,
                  max_position_pct: float = 0.25, period: str = DEFAULT_PERIOD) -> dict:
    """How many units the vol target allows, and what that risks."""
    ticker = ticker.strip().upper()
    frame = _prices(ticker, period)
    if frame.empty:
        return {"ticker": ticker, "error": "No price history could be loaded."}

    prepared = strat.prepare(_completed(frame, ticker))
    price = float(prepared.df["Close"].iloc[-1])
    atr = float(prepared.df["ATR"].iloc[-1])
    ann_vol = sizing_mod.annualized_vol_from_atr(atr, price)
    params = sizing_mod.SizingParams(target_vol=target_vol,
                                     max_position_pct=max_position_pct)
    quantity = sizing_mod.target_quantity(
        equity, price, ann_vol, strat.AssetClass.infer(ticker), params)
    notional = quantity * price

    # Did the cap bind, or the vol target? Comparing the *post*-cap notional to
    # the cap gets this wrong, because flooring to whole shares leaves it a few
    # dollars short of the limit it was actually clipped to.
    clamped_vol = min(max(ann_vol, params.min_vol), params.max_vol) if ann_vol > 0 else 0.0
    wanted = equity * target_vol / clamped_vol if clamped_vol > 0 else 0.0

    return {
        "ticker": ticker, "error": "", "price": price, "atr": atr,
        "annualised_vol": ann_vol, "quantity": quantity, "notional": notional,
        "pct_of_equity": notional / equity if equity else 0.0,
        "uncapped_notional": wanted,
        "capped": wanted > equity * max_position_pct,
        "equity": equity, "target_vol": target_vol,
    }


def render_size(info: dict) -> str:
    if info.get("error"):
        return f"Sizing for {info['ticker']}: {info['error']}"
    note = " (position cap binding)" if info["capped"] else ""
    return (f"Sizing {info['ticker']} at {info['price']:,.2f} with ATR {info['atr']:,.2f} "
            f"= {info['annualised_vol']:.0%} annualised volatility. At a "
            f"{info['target_vol']:.0%} vol target on ${info['equity']:,.0f}: "
            f"{info['quantity']:,.4f} units, ${info['notional']:,.0f}, "
            f"{info['pct_of_equity']:.1%} of equity{note}.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Random-entry test — the honesty tool
# ─────────────────────────────────────────────────────────────────────────────

def random_entry_test(ticker: str, period: str = DEFAULT_PERIOD,
                      n_paths: int = 500) -> dict:
    """Does the breakout rule beat entering at random for the same holding periods?

    A percentile near 50 means the entry rule added nothing over simply being in
    the market that long — the returns came from exposure, not selection. This is
    the finding a language model will never volunteer on its own.
    """
    ticker = ticker.strip().upper()
    frame = _prices(ticker, period)
    if frame.empty:
        return {"ticker": ticker, "error": "No price history could be loaded."}

    bench = _prices("^GSPC", period)
    benchmark = bench["Close"] if not bench.empty else None
    price_data = {ticker: frame}
    try:
        result = run_backtest(price_data, benchmark,
                              BacktestConfig(use_regime_gate=benchmark is not None))
    except Exception as exc:
        return {"ticker": ticker, "error": f"Backtest failed: {exc}"}

    if result.trades.empty:
        return {"ticker": ticker, "error": "",
                "n_trades": 0, "percentile": None,
                "note": "The breakout rule took no trades on this symbol, so there is "
                        "nothing to compare against random entry."}

    bench_result = sim.random_entry_benchmark(
        price_data, result.trades, sim.SimulationParams(n_paths=int(n_paths)))
    metrics = result.metrics
    return {
        "ticker": ticker, "error": "", "period": period,
        "n_trades": int(metrics["n_trades"]),
        "total_return": metrics["total_return"],
        "win_rate": metrics["win_rate"],
        "max_drawdown": metrics["max_drawdown"],
        "percentile": bench_result["percentile"] if bench_result else None,
        "actual_mean_trade": bench_result["actual_mean_trade"] if bench_result else None,
        "random_mean_trade": bench_result["random_mean_trade"] if bench_result else None,
        "note": "",
    }


def render_random_entry(info: dict) -> str:
    if info.get("error"):
        return f"Random-entry test for {info['ticker']}: {info['error']}"
    if not info["n_trades"]:
        return f"Random-entry test for {info['ticker']}: {info['note']}"

    pct = info["percentile"]
    if pct is None:
        verdict = "not enough trades to place it in a distribution"
    elif pct >= 90:
        verdict = "the entry rule is doing real work here"
    elif pct >= 60:
        verdict = "weak evidence the rule adds anything beyond exposure"
    else:
        verdict = ("the rule is NOT beating random entry of the same length — on this "
                   "symbol the returns came from being in the market, not from selection")
    return (
        f"Breakout rule on {info['ticker']} over {info['period']}: {info['n_trades']} trades, "
        f"total return {info['total_return']:+.1%}, win rate {info['win_rate']:.0%}, "
        f"max drawdown {info['max_drawdown']:.1%}. Average trade "
        f"{info['actual_mean_trade']:+.2%} against a random-entry average of "
        f"{info['random_mean_trade']:+.2%} — "
        + (f"{pct:.0f}th percentile. {verdict}." if pct is not None else f"{verdict}.")
        + " A single symbol is a characterisation, not an edge estimate."
    )
