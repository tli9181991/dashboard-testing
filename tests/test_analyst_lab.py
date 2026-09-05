"""Tests for the analyst test bench.

The harness is itself a thing that can be wrong. A scenario that quietly stops
exercising the branch it was written for still shows green, so these check that
the scenarios cover the branches they claim to, that the invariants actually fail
on a broken plan rather than only passing on a good one, and that the chart draws
the plan it was handed.
"""

from __future__ import annotations

import dataclasses

import plotly.graph_objects as go
import pytest

import analyst
import analyst_lab as lab


@pytest.fixture(scope="module")
def results():
    return lab.run_all()


# ── the scenarios ───────────────────────────────────────────────────────────

def test_every_scenario_behaves_as_documented(results):
    failures = [f"{s.key}: {e.detail}" for s, _, e in results if not e.ok]
    assert not failures, "scenarios no longer match their documented verdicts: " + \
                         "; ".join(failures)


def test_scenarios_cover_every_stance(results):
    """The harness is only worth running if it reaches every branch."""
    stances = {a.plan.stance for _, a, _ in results if a.ok}
    assert stances == {"buy now", "wait for the pullback", "stand aside"}


def test_scenarios_cover_the_distinct_refusals(results):
    """Each way of saying no needs its own case, or a regression hides in the gap."""
    blockers = " ".join(b for _, a, _ in results if a.ok for b in a.plan.blockers)
    assert "risk-off" in blockers
    assert "stage" in blockers
    assert "R away" in blockers
    assert any(not a.ok for _, a, _ in results), "no scenario covers the error path"


def test_scenarios_are_deterministic():
    first = {s.key: a.to_dict() for s, a, _ in lab.run_all()}
    second = {s.key: a.to_dict() for s, a, _ in lab.run_all()}
    assert first == second


def test_scenarios_run_without_network_or_model(results):
    """No scenario may reach for news, fundamentals or an LLM."""
    for _, analysis, _ in results:
        assert analysis.news_text == ""
        assert analysis.fundamentals_text == ""


def test_a_pullback_entry_is_anchored_below_the_current_price(results):
    """Anchoring above the price would be buying a bounce into resistance."""
    waits = [a for _, a, _ in results if a.ok and a.plan.stance == "wait for the pullback"]
    assert waits, "no scenario exercises the pullback branch"
    for analysis in waits:
        assert analysis.plan.entry_high < analysis.trend.price


def test_expectation_check_detects_a_wrong_verdict():
    """The checker must be able to fail, or the green row means nothing."""
    scenario = lab.SCENARIOS_BY_KEY["downtrend"]
    analysis = lab.run_scenario(scenario)
    impossible = dataclasses.replace(scenario, expect_stance="buy now")
    assert not lab.check_expectation(impossible, analysis).ok


def test_parameters_flow_through_to_the_scenarios():
    """Relaxing the reward:risk floor must un-block the scenario refused on it."""
    scenario = lab.SCENARIOS_BY_KEY["target_too_close"]
    assert lab.run_scenario(scenario).plan.blockers

    lenient = analyst.AnalystParams(min_reward_risk=0.1)
    assert not lab.run_scenario(scenario, lenient).plan.blockers


def test_a_projected_target_is_not_reported_as_a_verified_reward_risk():
    """T1 projected at a chosen multiple cannot also be evidence the trade clears it."""
    analysis = lab.run_scenario(lab.SCENARIOS_BY_KEY["advancing"])
    assert "projection" in analysis.plan.target1_source
    assert not any("R away" in b for b in analysis.plan.blockers)
    assert any("chosen, not verified" in n for n in analysis.plan.notes)

    detail = next(c.detail for c in lab.plan_invariants(analysis)
                  if c.label == "reward:risk meets the minimum")
    assert "chosen, not verified" in detail


# ── the invariants ──────────────────────────────────────────────────────────

def test_invariants_pass_on_the_tradeable_scenario():
    analysis = lab.run_scenario(lab.SCENARIOS_BY_KEY["advancing"])
    checks = lab.plan_invariants(analysis)
    assert checks and all(c.ok for c in checks), \
        [f"{c.label}: {c.detail}" for c in checks if not c.ok]


def test_invariants_catch_a_corrupted_plan():
    """Prove the checks bite: move the stop above the entry and they must fail."""
    analysis = lab.run_scenario(lab.SCENARIOS_BY_KEY["advancing"])
    broken = dataclasses.replace(
        analysis, plan=dataclasses.replace(analysis.plan,
                                           stop=analysis.plan.entry_high + 1.0))
    failed = [c.label for c in lab.plan_invariants(broken) if not c.ok]
    assert "stop below entry" in failed


def test_invariants_catch_an_inconsistent_size():
    analysis = lab.run_scenario(lab.SCENARIOS_BY_KEY["advancing"])
    broken = dataclasses.replace(
        analysis, plan=dataclasses.replace(analysis.plan, dollar_risk=1.0))
    failed = [c.label for c in lab.plan_invariants(broken) if not c.ok]
    assert "dollar risk matches size × stop distance" in failed


def test_a_refusal_has_its_own_invariants():
    analysis = lab.run_scenario(lab.SCENARIOS_BY_KEY["downtrend"])
    checks = lab.plan_invariants(analysis)
    assert all(c.ok for c in checks)
    assert {c.label for c in checks} == {"no entry names no price",
                                         "refusal states a reason"}


def test_invariants_report_an_errored_analysis():
    analysis = lab.run_scenario(lab.SCENARIOS_BY_KEY["short_history"])
    checks = lab.plan_invariants(analysis)
    assert not checks[0].ok


# ── the chart ───────────────────────────────────────────────────────────────

def test_chart_draws_the_plan_levels():
    scenario = lab.SCENARIOS_BY_KEY["advancing"]
    analysis = lab.run_scenario(scenario)
    fig = lab.build_plan_chart(analysis, scenario.prices())
    assert isinstance(fig, go.Figure)

    drawn = [s.y0 for s in fig.layout.shapes]
    for level in (analysis.plan.stop, analysis.plan.target1, analysis.plan.target2):
        assert any(abs(y - level) < 1e-6 for y in drawn if y is not None), \
            f"{level} missing from the chart"


def test_chart_y_range_contains_every_plan_level():
    """A target drawn outside the visible range is a chart that lies by omission."""
    scenario = lab.SCENARIOS_BY_KEY["advancing"]
    analysis = lab.run_scenario(scenario)
    fig = lab.build_plan_chart(analysis, scenario.prices())
    low, high = fig.layout.yaxis.range
    for level in (analysis.plan.stop, analysis.plan.entry_low,
                  analysis.plan.entry_high, analysis.plan.target1, analysis.plan.target2):
        assert low <= level <= high


def test_chart_renders_for_a_refusal():
    """No entry means no plan lines, but the chart must still draw."""
    scenario = lab.SCENARIOS_BY_KEY["downtrend"]
    analysis = lab.run_scenario(scenario)
    fig = lab.build_plan_chart(analysis, scenario.prices())
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.range[0] < analysis.trend.price < fig.layout.yaxis.range[1]
