"""Streamlit test bench for ``analyst``.

    streamlit run analyst_app.py

Deliberately separate from ``app.py``: this exists to *exercise* the analyst, not
to trade from it, and coupling it to the 2,000-line dashboard would mean the test
bench breaks whenever the dashboard does.

It opens on the scenario harness rather than on a ticker box, which is the whole
argument for the app. Pointing an advisor at whatever the market is doing today
exercises exactly one of its paths; the other five — the risk-off veto, the
refusal on reward:risk, the pullback anchor, the short-history error — only show
up on days you cannot schedule. Every scenario here is synthetic, deterministic
and offline, carries the verdict it is supposed to produce, and can be run
against any parameters you like.

The live tab does the same thing against a real symbol, using the cache the IB
notebook writes to, so the two paths differ only in where the bars came from.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import analyst
import analyst_lab as lab
import data as data_mod

st.set_page_config(page_title="Analyst test bench", page_icon="🔬", layout="wide")

STANCE_STYLE = {
    "buy now": ("🟢", "success"),
    "wait for the pullback": ("🟡", "warning"),
    "stand aside": ("🔴", "error"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — the parameters every panel runs under
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_params() -> analyst.AnalystParams:
    st.sidebar.header("Parameters")
    st.sidebar.caption(
        "These are the knobs the analyst runs under. Change one and every panel "
        "recomputes, which is the quickest way to see what a threshold is actually "
        "doing to the output.")
    equity = st.sidebar.number_input("Account equity ($)", 1_000.0, 100_000_000.0,
                                     float(analyst.AnalystParams.equity), step=5_000.0)
    target_vol = st.sidebar.slider("Target volatility", 0.05, 0.50,
                                   analyst.AnalystParams.target_vol, 0.01)
    max_pos = st.sidebar.slider("Max position (% of equity)", 0.05, 1.0,
                                analyst.AnalystParams.max_position_pct, 0.05)
    min_rr = st.sidebar.slider("Minimum reward:risk", 0.5, 5.0,
                               analyst.AnalystParams.min_reward_risk, 0.25)
    max_risk_atr = st.sidebar.slider("Max stop distance (ATR)", 0.5, 6.0,
                                     analyst.AnalystParams.max_risk_atr, 0.25)
    return analyst.AnalystParams(
        equity=equity, target_vol=target_vol, max_position_pct=max_pos,
        min_reward_risk=min_rr, max_risk_atr=max_risk_atr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared rendering
# ─────────────────────────────────────────────────────────────────────────────

def show_verdict(analysis: analyst.StockAnalysis) -> None:
    plan = analysis.plan
    icon, kind = STANCE_STYLE.get(plan.stance, ("⚪", "info"))
    getattr(st, kind)(f"{icon}  **{plan.stance.upper()}**"
                      + (f" — engine signal {plan.action}" if plan.action else ""))
    for blocker in plan.blockers:
        st.markdown(f"- ✗ {blocker}")


def show_plan(analysis: analyst.StockAnalysis) -> None:
    plan, trend = analysis.plan, analysis.trend
    if plan.entry_low is None:
        st.info("No entry. There is no buy price worth naming here — which is a "
                "result, not a gap.")
        return

    if plan.blockers:
        st.warning("Reference only — the blockers above mean this is not a trade "
                   "today. These are the levels that would apply if they cleared.")

    a, b, c, d = st.columns(4)
    a.metric("Buy zone", f"{plan.entry_low:,.2f} – {plan.entry_high:,.2f}")
    b.metric("Stop", f"{plan.stop:,.2f}", f"-{plan.risk_per_share:,.2f}/share",
             delta_color="inverse")
    c.metric("Target 1", f"{plan.target1:,.2f}", f"{plan.reward_risk_1:.2f}R")
    d.metric("Target 2", f"{plan.target2:,.2f}", f"{plan.reward_risk_2:.2f}R")

    e, f, g = st.columns(3)
    e.metric("Size", f"{plan.quantity:,.0f} units", f"${plan.notional:,.0f}")
    f.metric("At risk", f"${plan.dollar_risk:,.0f}",
             f"{plan.risk_pct_of_equity:.2%} of equity")
    g.metric("Trailing exit", f"{plan.trailing_exit:,.2f}", "sells on a close below")

    st.caption("**Where the numbers come from.** " + "  ".join(f"· {n}" for n in plan.notes))
    st.caption(f"Target 1: {plan.target1_source}. Target 2: {plan.target2_source}.")


def show_trend(analysis: analyst.StockAnalysis) -> None:
    t = analysis.trend
    a, b, c, d = st.columns(4)
    a.metric("Price", f"{t.price:,.2f}")
    b.metric("Trend stage", t.stage)
    c.metric("ATR", f"{t.atr:,.2f}", f"{t.atr_pct:.1%} of price")
    d.metric("From 52w high", f"{t.pct_from_high:+.1%}")
    a2, b2, c2, d2 = st.columns(4)
    a2.metric("20 EMA", f"{t.ema20:,.2f}")
    b2.metric("50 EMA", f"{t.ema50:,.2f}")
    c2.metric("200 EMA", f"{t.ema200:,.2f}", f"{t.ema200_slope:+.2%} / 20 sessions")
    d2.metric("Regime", "n/a" if t.regime_ok is None
              else ("risk-on" if t.regime_ok else "RISK-OFF"))
    st.caption(t.stage_detail)


def show_invariants(analysis: analyst.StockAnalysis,
                    params: analyst.AnalystParams) -> None:
    """The guarantees, checked live rather than asserted in a docstring."""
    rows = lab.plan_invariants(analysis, params)
    st.dataframe(
        pd.DataFrame([{"": "✓" if r.ok else "✗", "Invariant": r.label,
                       "Detail": r.detail} for r in rows]),
        hide_index=True, use_container_width=True)
    broken = [r.label for r in rows if not r.ok]
    if broken:
        st.error("Broken: " + ", ".join(broken))


def show_long_term(analysis: analyst.StockAnalysis) -> None:
    view = analysis.long_term
    st.subheader(f"Long-term hold — {view.verdict}")
    st.caption(f"{view.passed} of {view.known} known criteria pass"
               + (f", {len(view.unknowns)} unknown" if view.unknowns else "")
               + f" · basis: {view.basis}")
    if view.price_only:
        st.warning("Nothing about the business was available — this is a read on the "
                   "chart, not on the company.")
    mark = {"pass": "✓", "fail": "✗", "unknown": "?"}
    st.dataframe(
        pd.DataFrame([{"": mark[c.state], "Criterion": c.label, "Detail": c.detail}
                      for c in view.criteria]),
        hide_index=True, use_container_width=True)


def show_analysis(analysis: analyst.StockAnalysis, prices: pd.DataFrame,
                  params: analyst.AnalystParams) -> None:
    if not analysis.ok:
        st.error(f"{analysis.ticker}: {analysis.error}")
        return

    show_verdict(analysis)
    st.divider()
    show_trend(analysis)
    st.divider()
    st.subheader("Trade plan")
    show_plan(analysis)
    st.plotly_chart(lab.build_plan_chart(analysis, prices), use_container_width=True)
    st.divider()
    show_long_term(analysis)
    st.divider()
    st.subheader("Invariants")
    st.caption("Recomputed on this result. These are the relationships that have to "
               "hold for the plan to mean anything.")
    show_invariants(analysis, params)

    if analysis.news_text:
        with st.expander("News"):
            st.text(analysis.news_text)
    if analysis.fundamentals_text:
        with st.expander("Fundamentals"):
            st.text(analysis.fundamentals_text)
    with st.expander("Plain-text report (what the model is given)"):
        st.code(analysis.render(), language="text")
    with st.expander("Raw analysis (JSON)"):
        st.code(json.dumps(analysis.to_dict(), indent=2, default=str), language="json")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

def scenario_tab(params: analyst.AnalystParams) -> None:
    st.subheader("Scenario harness")
    st.caption(
        "Synthetic, deterministic markets that drive every branch of the analyst. "
        "Each carries the verdict it is supposed to reach, so this is a regression "
        "check you can read, not a demo.")

    results = lab.run_all(params)
    passed = sum(1 for _, _, e in results if e.ok)
    summary = pd.DataFrame([
        {"": "✓" if e.ok else "✗", "Scenario": s.label,
         "Expected": s.expect_stance or ("an error" if s.expect_error else "—"),
         "Got": e.detail}
        for s, _, e in results])
    (st.success if passed == len(results) else st.error)(
        f"{passed} of {len(results)} scenarios behave as documented"
        + ("" if passed == len(results) else " — see the ✗ rows below"))
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption("A ✗ here is not necessarily a bug in the analyst — changing a "
               "parameter in the sidebar can legitimately change a verdict. It means "
               "the behaviour no longer matches what the scenario documents.")

    st.divider()
    labels = {s.label: s for s, _, _ in results}
    chosen = labels[st.selectbox("Inspect a scenario", list(labels))]
    st.info(chosen.description)
    analysis = lab.run_scenario(chosen, params)
    show_analysis(analysis, chosen.prices(), params)


def live_tab(params: analyst.AnalystParams) -> None:
    st.subheader("Live symbol")
    left, right = st.columns([2, 3])
    symbol = left.text_input("Symbol", "AAPL").strip().upper()
    period = right.selectbox("History", ["1y", "2y", "3y", "5y"], index=2)
    with_news = left.checkbox("Fetch news", value=False,
                              help="Needs network, and Azure credentials to be scored.")
    with_fundamentals = right.checkbox("Fetch fundamentals", value=False,
                                       help="Needs network.")
    st.caption("Reads `.screen_cache/` first, so bars written by "
               "`notebooks/ib_tws_connect.ipynb` are used in preference to yfinance.")

    if not st.button("Analyse", type="primary"):
        return

    run_params = analyst.AnalystParams(
        equity=params.equity, target_vol=params.target_vol,
        max_position_pct=params.max_position_pct, min_reward_risk=params.min_reward_risk,
        max_risk_atr=params.max_risk_atr, period=period)
    with st.spinner(f"Analysing {symbol}…"):
        analysis = analyst.analyze(symbol, run_params, with_news=with_news,
                                   with_fundamentals=with_fundamentals)
        prices = data_mod.load_history(symbol, period=period)
    if prices.empty:
        st.error(f"No price history for {symbol}. Nothing to draw.")
        return
    show_analysis(analysis, prices, run_params)

    if st.button("Have the model write it up"):
        with st.spinner("Narrating…"):
            st.markdown(analyst.narrate(analysis))


def narration_tab(params: analyst.AnalystParams) -> None:
    st.subheader("Narration check")
    st.caption(
        "The analyst's one rule is that the model narrates and never computes. This "
        "runs a scenario, shows the report the model is given, and puts its prose "
        "beside the computed numbers so you can check that no new price appeared.")

    labels = {s.label: s for s in lab.SCENARIOS if not s.expect_error}
    chosen = labels[st.selectbox("Scenario", list(labels))]
    analysis = lab.run_scenario(chosen, params)
    plan = analysis.plan

    if plan.entry_low is not None:
        st.caption("Every number the model is allowed to state:")
        st.code(
            f"buy zone   {plan.entry_low:,.2f} – {plan.entry_high:,.2f}\n"
            f"stop       {plan.stop:,.2f}\n"
            f"target 1   {plan.target1:,.2f}\n"
            f"target 2   {plan.target2:,.2f}\n"
            f"size       {plan.quantity:,.0f} units, ${plan.notional:,.0f}",
            language="text")

    if st.button("Narrate", type="primary"):
        with st.spinner("Narrating…"):
            text = analyst.narrate(analysis)
        if text == analysis.render():
            st.info("No Azure credentials configured, so narration fell back to the "
                    "computed report. The plan is unaffected — that is the point of "
                    "the split.")
        st.markdown(text)


def main() -> None:
    st.title("🔬 Analyst test bench")
    st.caption("Exercises `analyst.py` — the trade plan, the long-term verdict, and "
               "the rule that the model never computes a number.")
    params = sidebar_params()
    scenarios, live, narration = st.tabs(
        ["Scenarios", "Live symbol", "Narration"])
    with scenarios:
        scenario_tab(params)
    with live:
        live_tab(params)
    with narration:
        narration_tab(params)


if __name__ == "__main__":
    main()
