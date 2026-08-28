"""Behaviour of the two trading rules, including the bugs this refactor fixed."""

import numpy as np
import pandas as pd
import pytest

from strategy import (
    Action,
    AssetClass,
    Position,
    StrategyParams,
    drop_forming_bar,
    evaluate,
    prepare,
)


def _frame_below_sma(n=60):
    """Build a history that rises, then rolls over so the last close is under the SMA."""
    up = np.linspace(100, 140, n - 10)
    down = np.linspace(140, 118, 10)
    close = np.concatenate([up, down])
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_long_term_flag_never_produces_a_buy_below_the_exit_average():
    """Regression: the original code fell through to the entry block when long_term was set."""
    frame = prepare(_frame_below_sma())
    last = len(frame) - 1
    assert frame.df["Close"].iloc[last] < frame.df["SMA_10"].iloc[last]

    decision = evaluate(frame, last, Position(quantity=5, avg_price=100.0, long_term=True))
    assert decision.action is Action.HOLD
    assert any("long-term" in log for log in decision.logs)


def test_ordinary_holding_below_the_exit_average_sells():
    frame = prepare(_frame_below_sma())
    last = len(frame) - 1
    decision = evaluate(frame, last, Position(quantity=5, avg_price=100.0, long_term=False))
    assert decision.action is Action.SELL


def test_flat_below_the_exit_average_does_nothing():
    frame = prepare(_frame_below_sma())
    decision = evaluate(frame, len(frame) - 1, Position())
    assert decision.action is Action.HOLD


def test_regime_veto_blocks_entries(prices):
    """Any BUY the strategy would take must become HOLD when the gate is shut."""
    frame = prepare(prices)
    vetoed = 0
    for i in range(250, len(frame)):
        allowed = evaluate(frame, i, Position(), regime_ok=True)
        if allowed.action is Action.BUY:
            blocked = evaluate(frame, i, Position(), regime_ok=False)
            assert blocked.action is Action.HOLD
            vetoed += 1
    assert vetoed > 0, "fixture produced no entries; test proves nothing"


def test_regime_veto_never_blocks_an_exit():
    frame = prepare(_frame_below_sma())
    last = len(frame) - 1
    held = Position(quantity=5, avg_price=100.0)
    assert evaluate(frame, last, held, regime_ok=False).action is Action.SELL


def test_no_pyramiding_into_an_open_position(prices):
    frame = prepare(prices)
    for i in range(250, len(frame)):
        if evaluate(frame, i, Position(), regime_ok=True).action is Action.BUY:
            held = Position(quantity=10, avg_price=float(frame.df["Close"].iloc[i]))
            assert evaluate(frame, i, held, regime_ok=True).action is not Action.BUY


def test_asset_class_inference():
    assert AssetClass.infer("ETH-USD") is AssetClass.CRYPTO
    assert AssetClass.infer("btc-usd") is AssetClass.CRYPTO
    assert AssetClass.infer("NFLX") is AssetClass.EQUITY


def test_drop_forming_bar_removes_todays_incomplete_candle(prices):
    now = pd.Timestamp(prices.index[-1])
    trimmed = drop_forming_bar(prices, AssetClass.EQUITY, now=now)
    assert len(trimmed) == len(prices) - 1
    assert trimmed.index[-1] == prices.index[-2]


def test_drop_forming_bar_keeps_a_closed_final_bar(prices):
    later = pd.Timestamp(prices.index[-1]) + pd.Timedelta(days=3)
    assert len(drop_forming_bar(prices, AssetClass.EQUITY, now=later)) == len(prices)


def test_insufficient_history_holds():
    idx = pd.bdate_range("2023-01-02", periods=5)
    tiny = pd.DataFrame(
        {"Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5,
         "Close": [1.0] * 5, "Volume": [1.0] * 5},
        index=idx,
    )
    frame = prepare(tiny)
    assert evaluate(frame, len(frame) - 1, Position()).action is Action.HOLD
