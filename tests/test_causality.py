"""The load-bearing tests: a decision must never change when the future arrives.

If these fail, every backtest number produced by this repo is fiction.
"""

import pandas as pd
import pytest

from strategy import Position, StrategyParams, detect_levels, evaluate, prepare, add_indicators


def test_levels_are_never_revised_by_later_bars(prices):
    """A level visible at bar i must be identical whether or not later bars exist."""
    params = StrategyParams()
    full = detect_levels(add_indicators(prices, params), params)

    for cutoff in (250, 350, 450, 599):
        truncated = detect_levels(add_indicators(prices.iloc[: cutoff + 1], params), params)
        visible_full = [lv for lv in full if lv.confirmed_at <= cutoff]
        visible_trunc = [lv for lv in truncated if lv.confirmed_at <= cutoff]
        assert visible_full == visible_trunc, f"levels were revised at cutoff {cutoff}"


def test_decision_is_identical_on_truncated_history(prices):
    """Replaying with only the bars available at the time must reproduce the decision."""
    params = StrategyParams()
    full_frame = prepare(prices, params)

    for cutoff in range(220, 600, 17):
        trunc_frame = prepare(prices.iloc[: cutoff + 1], params)
        for position in (Position(), Position(quantity=10, avg_price=100.0)):
            a = evaluate(full_frame, cutoff, position)
            b = evaluate(trunc_frame, cutoff, position)
            assert a.action == b.action, f"action diverged at bar {cutoff}"
            assert a.price == pytest.approx(b.price)
            assert a.sma_exit == pytest.approx(b.sma_exit, nan_ok=True)
            assert a.atr == pytest.approx(b.atr)
            assert a.broken_resistance == pytest.approx(b.broken_resistance, nan_ok=True)
            assert a.next_resistance == pytest.approx(b.next_resistance, nan_ok=True)


def test_appending_a_bar_does_not_rewrite_history(prices):
    """Growing the frame one bar at a time leaves prior decisions untouched."""
    params = StrategyParams()
    target = 400
    decisions = []
    for extra in range(0, 40, 8):
        frame = prepare(prices.iloc[: target + 1 + extra], params)
        decisions.append(evaluate(frame, target, Position()))

    first = decisions[0]
    for later in decisions[1:]:
        assert later.action == first.action
        assert later.price == pytest.approx(first.price)
        assert later.broken_resistance == pytest.approx(first.broken_resistance, nan_ok=True)


def test_indicators_use_no_future_data(prices):
    params = StrategyParams()
    full = add_indicators(prices, params)
    cutoff = 400
    trunc = add_indicators(prices.iloc[: cutoff + 1], params)

    for col in ("SMA_10", "ATR", "EMA_5", "EMA_20", "EMA_200"):
        assert full[col].iloc[cutoff] == pytest.approx(trunc[col].iloc[cutoff]), col
