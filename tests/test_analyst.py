"""Tests for the single-symbol analyst.

The concern specific to this module is that it emits *prices* — an entry, a stop,
two targets — and a plan whose numbers are subtly wrong is worse than no plan.
So these assert the arithmetic relationships that have to hold for the output to
mean anything (stop below entry, targets above it, risk inside the budget), and
that a setup which fails a rule is reported as a refusal rather than softened.
"""

from __future__ import annotations

import pandas as pd
import pytest

import analyst
import data as data_mod


@pytest.fixture(scope="module")
def uptrend():
    """A synthetic series that ends in a clean advance."""
    return data_mod.synthetic_ohlcv(n=600, seed=3, annual_drift=0.18, annual_vol=0.28)


@pytest.fixture(scope="module")
def downtrend():
    return data_mod.synthetic_ohlcv(n=600, seed=42)


@pytest.fixture(scope="module")
def risk_on(uptrend):
    """A benchmark holding above its own 200 SMA, so the regime gate opens."""
    return data_mod.synthetic_benchmark(uptrend.index, seed=3, annual_drift=0.25,
                                        annual_vol=0.12)


@pytest.fixture(scope="module")
def risk_off(uptrend):
    return data_mod.synthetic_benchmark(uptrend.index, seed=5)


def _analyze(prices, bench):
    return analyst.analyze("TEST", prices=prices, benchmark_close=bench,
                           with_news=False, with_fundamentals=False)


# ── the plan ────────────────────────────────────────────────────────────────

def test_advance_in_a_risk_on_market_produces_an_actionable_plan(uptrend, risk_on):
    analysis = _analyze(uptrend, risk_on)
    assert analysis.ok
    assert analysis.trend.stage == "advancing"
    assert analysis.plan.actionable
    assert analysis.plan.blockers == ()


def test_plan_prices_are_strictly_ordered(uptrend, risk_on):
    """stop < entry <= entry_high < target1 < target2, or the plan is nonsense."""
    plan = _analyze(uptrend, risk_on).plan
    assert plan.stop < plan.entry_low <= plan.entry_high
    assert plan.entry_high < plan.target1 < plan.target2


def test_risk_and_size_are_internally_consistent(uptrend, risk_on):
    plan = _analyze(uptrend, risk_on).plan
    entry_mid = (plan.entry_low + plan.entry_high) / 2
    assert plan.risk_per_share == pytest.approx(entry_mid - plan.stop)
    assert plan.dollar_risk == pytest.approx(plan.quantity * plan.risk_per_share)
    assert plan.notional == pytest.approx(plan.quantity * entry_mid)
    assert plan.reward_risk_1 == pytest.approx(
        (plan.target1 - entry_mid) / plan.risk_per_share)


def test_position_never_exceeds_the_equity_cap(uptrend, risk_on):
    params = analyst.AnalystParams(equity=50_000.0, max_position_pct=0.25)
    plan = analyst.analyze("TEST", params, prices=uptrend, benchmark_close=risk_on,
                           with_news=False, with_fundamentals=False).plan
    assert plan.notional <= params.equity * params.max_position_pct


@pytest.mark.parametrize("seed", [3, 13, 21, 24, 31, 37])
def test_stop_is_never_further_than_the_atr_budget(seed, risk_on):
    """A stop wider than the budget is not caution, it is an unsizeable position."""
    prices = data_mod.synthetic_ohlcv(n=600, seed=seed, annual_drift=0.18, annual_vol=0.28)
    bench = data_mod.synthetic_benchmark(prices.index, seed=3, annual_drift=0.25,
                                         annual_vol=0.12)
    analysis = _analyze(prices, bench)
    plan = analysis.plan
    if plan.entry_low is None:
        pytest.skip("no entry proposed for this series")
    entry_mid = (plan.entry_low + plan.entry_high) / 2
    budget = analyst.AnalystParams().max_risk_atr * analysis.trend.atr
    assert entry_mid - plan.stop <= budget + 1e-6


def test_a_target_too_close_to_pay_for_the_stop_is_blocked(risk_on):
    """Seed 21 puts resistance under 1R away. That must read as a refusal."""
    prices = data_mod.synthetic_ohlcv(n=600, seed=21, annual_drift=0.18, annual_vol=0.28)
    bench = data_mod.synthetic_benchmark(prices.index, seed=3, annual_drift=0.25,
                                         annual_vol=0.12)
    plan = _analyze(prices, bench).plan
    assert plan.reward_risk_1 < analyst.AnalystParams().min_reward_risk
    assert any("R away" in b for b in plan.blockers)
    assert not plan.actionable
    assert plan.stance == "stand aside"


# ── refusals ────────────────────────────────────────────────────────────────

def test_downtrend_names_no_buy_price(downtrend, risk_on):
    analysis = _analyze(downtrend, risk_on)
    assert analysis.trend.stage in ("declining", "repairing")
    assert analysis.plan.entry_low is None
    assert analysis.plan.stop is None
    assert analysis.plan.blockers
    assert "No entry" in analysis.render()


def test_risk_off_regime_vetoes_a_new_entry(uptrend, risk_off):
    analysis = _analyze(uptrend, risk_off)
    assert analysis.trend.regime_ok is False
    assert not analysis.plan.actionable
    assert any("risk-off" in b for b in analysis.plan.blockers)


def test_short_history_is_reported_not_guessed():
    short = data_mod.synthetic_ohlcv(n=120, seed=1)
    analysis = analyst.analyze("TEST", prices=short, benchmark_close=None,
                               with_news=False, with_fundamentals=False)
    assert not analysis.ok
    assert "200 EMA" in analysis.error
    assert analysis.plan is None


# ── the long-term question ──────────────────────────────────────────────────

def test_missing_fundamentals_count_as_unknown_not_as_failures(uptrend, risk_on):
    view = _analyze(uptrend, risk_on).long_term
    assert view.known == 4, "only the four price-based criteria are knowable here"
    assert set(view.unknowns) == {
        "Profitable", "Revenue growing",
        "Balance sheet covers near-term liabilities", "News not negative",
    }
    assert view.passed <= view.known


def test_long_term_verdict_splits_from_the_trade_verdict(uptrend, risk_off):
    """Risk-off blocks the trade; it must not by itself condemn the holding."""
    analysis = _analyze(uptrend, risk_off)
    assert not analysis.plan.actionable
    assert analysis.long_term.verdict in ("ACCUMULATE", "HOLD")


def test_downtrend_long_term_verdict_is_negative(downtrend, risk_on):
    assert _analyze(downtrend, risk_on).long_term.verdict in ("REDUCE", "EXIT")


def test_a_verdict_with_no_business_data_says_it_is_price_only(uptrend, risk_on):
    """"Hold for the long term" reached from moving averages alone must admit it."""
    analysis = _analyze(uptrend, risk_on)
    view = analysis.long_term
    assert view.price_only
    assert view.basis == "price only"
    assert "read on the chart, not on the company" in analysis.render()


def test_one_business_criterion_lifts_the_verdict_off_price_alone(uptrend, risk_on):
    analysis = _analyze(uptrend, risk_on)
    # Same trend, but with a news reading available.
    bench = analyst._aligned_benchmark(risk_on, analysis.trend and uptrend.index)
    view = analyst.long_term_view(analysis.trend, uptrend["Close"], bench,
                                  None, "positive", analyst.AnalystParams())
    assert not view.price_only
    assert view.basis == "price and business"


# ── causality and purity ────────────────────────────────────────────────────

def test_future_bars_cannot_change_a_plan_already_given(uptrend, risk_on):
    """Analysis of the first k bars must not depend on what came after them.

    Level causality itself is proven upstream in ``tests/test_causality.py``; what
    this covers is the trend, level-selection and plan code added here, against
    deliberately extreme future bars.
    """
    k = 450
    head = uptrend.iloc[:k]
    shock = uptrend.iloc[k:].copy() * 5.0          # a violent future that never happened yet
    with_future = pd.concat([head, shock])

    baseline = _analyze(head, risk_on.iloc[:k])
    replayed = _analyze(with_future.iloc[:k], risk_on.iloc[:k])
    assert baseline.to_dict() == replayed.to_dict()


def test_analysis_is_deterministic(uptrend, risk_on):
    assert _analyze(uptrend, risk_on).to_dict() == _analyze(uptrend, risk_on).to_dict()


# ── output ──────────────────────────────────────────────────────────────────

def test_render_states_every_headline_number(uptrend, risk_on):
    analysis = _analyze(uptrend, risk_on)
    text = analysis.render()
    for heading in ("CURRENT STATE", "LEVELS", "TRADE PLAN", "LONG-TERM HOLD"):
        assert heading in text
    for value in (analysis.plan.entry_low, analysis.plan.stop,
                  analysis.plan.target1, analysis.plan.target2):
        assert f"{value:,.2f}" in text


def test_narration_falls_back_to_the_computed_report_without_credentials(
        monkeypatch, uptrend, risk_on):
    """No LLM must cost you the prose, never the plan."""
    monkeypatch.setattr("config.AZURE_INFERENCE_ENDPOINT", "", raising=False)
    monkeypatch.setattr("config.AZURE_INFERENCE_CREDENTIAL", "", raising=False)
    analysis = _analyze(uptrend, risk_on)
    assert analyst.narrate(analysis) == analysis.render()


def test_a_blocked_plan_marks_its_levels_reference_only(uptrend, risk_off):
    """A reader who takes the zone and skips the refusal is the failure to prevent."""
    analysis = _analyze(uptrend, risk_off)
    assert analysis.plan.blockers
    assert analysis.plan.entry_low is not None, "this case still computes levels"
    report = analysis.render()
    assert "REFERENCE ONLY" in report
    assert report.index("✗") < report.index("REFERENCE ONLY") < report.index("Buy zone")


def test_an_unblocked_plan_carries_no_reference_only_marker(uptrend, risk_on):
    assert "REFERENCE ONLY" not in _analyze(uptrend, risk_on).render()
