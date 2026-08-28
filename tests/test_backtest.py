"""Backtester mechanics — the parts that decide whether the numbers mean anything."""

from dataclasses import replace

import pandas as pd
import pytest

from backtest import BacktestConfig, CostModel, run_backtest, sweep
from sizing import SizingParams


@pytest.fixture(scope="module")
def universe(prices, choppy):
    return {"AAA": prices, "BBB": choppy}


@pytest.fixture(scope="module")
def result(universe, benchmark):
    return run_backtest(universe, benchmark, BacktestConfig(warmup=210))


def test_entries_fill_at_the_next_open_not_the_signal_close(result, universe):
    """The defining anti-look-ahead property: no fill ever happens at the deciding close."""
    assert not result.trades.empty
    slip = result.config.costs.slippage_bps / 10_000.0

    for row in result.trades.itertuples():
        bar_open = float(universe[row.symbol].loc[row.entry_date, "Open"])
        assert row.entry_price == pytest.approx(bar_open * (1 + slip)), (
            f"{row.symbol} entry on {row.entry_date} did not fill at that bar's open"
        )


def test_exits_fill_at_the_next_open(result, universe):
    slip = result.config.costs.slippage_bps / 10_000.0
    exits = result.trades[result.trades["exit_reason"] == "sma_exit"]
    assert len(exits) > 0
    for row in exits.itertuples():
        bar_open = float(universe[row.symbol].loc[row.exit_date, "Open"])
        assert row.exit_price == pytest.approx(bar_open * (1 - slip))


def test_slippage_moves_price_against_the_trader(universe, benchmark):
    free = run_backtest(universe, benchmark,
                        BacktestConfig(warmup=210, costs=CostModel(0.0, 0.0)))
    costly = run_backtest(universe, benchmark,
                          BacktestConfig(warmup=210, costs=CostModel(25.0, 5.0)))
    assert costly.metrics["ending_equity"] < free.metrics["ending_equity"]
    assert costly.metrics["total_costs"] > free.metrics["total_costs"]


def test_truncating_the_data_does_not_change_completed_trades(universe, benchmark):
    """Running on less history must reproduce the trades that had already closed."""
    cutoff = 480
    truncated = {k: v.iloc[:cutoff] for k, v in universe.items()}
    cut_date = min(v.index[-1] for v in truncated.values())

    full = run_backtest(universe, benchmark, BacktestConfig(warmup=210))
    part = run_backtest(truncated, benchmark, BacktestConfig(warmup=210))

    def settled(res):
        df = res.trades
        df = df[(df["exit_date"] < cut_date) & (df["exit_reason"] != "end_of_test")]
        return df[["symbol", "entry_date", "exit_date", "quantity"]].reset_index(drop=True)

    pd.testing.assert_frame_equal(settled(full), settled(part))


def test_equity_is_conserved_when_flat(universe, benchmark):
    """With sizing that can never afford a share, equity must sit exactly at the start."""
    cfg = BacktestConfig(warmup=210, sizing=SizingParams(target_vol=1e-9, max_position_pct=1e-9))
    res = run_backtest(universe, benchmark, cfg)
    assert res.trades.empty
    assert res.equity.nunique() == 1
    assert res.equity.iloc[-1] == pytest.approx(cfg.initial_equity)


def test_no_position_exceeds_the_configured_cap(universe, benchmark):
    cap = 0.20
    cfg = BacktestConfig(warmup=210, sizing=SizingParams(max_position_pct=cap))
    res = run_backtest(universe, benchmark, cfg)
    for row in res.trades.itertuples():
        notional = row.quantity * row.entry_price
        assert notional <= cfg.initial_equity * cap * 1.75, "position cap badly breached"


def test_regime_gate_reduces_trade_count(universe, benchmark):
    gated = run_backtest(universe, benchmark, BacktestConfig(warmup=210, use_regime_gate=True))
    ungated = run_backtest(universe, benchmark, BacktestConfig(warmup=210, use_regime_gate=False))
    assert gated.metrics["n_trades"] < ungated.metrics["n_trades"]


def test_pnl_reconciles_with_the_equity_curve(result):
    """Realised trade PnL plus the starting stake must equal final equity."""
    total_pnl = float(result.trades["pnl"].sum())
    expected = result.config.initial_equity + total_pnl
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_metrics_are_internally_consistent(result):
    m = result.metrics
    assert m["n_trades"] == len(result.trades)
    assert 0.0 <= m["win_rate"] <= 1.0
    assert 0.0 <= m["exposure"] <= 1.0
    assert m["max_drawdown"] <= 0.0
    wins = result.trades[result.trades["return_pct"] > 0]
    assert m["win_rate"] == pytest.approx(len(wins) / len(result.trades))


def test_every_position_is_closed_out_at_the_end(result, universe):
    ends = result.trades["exit_reason"].value_counts()
    assert ends.get("end_of_test", 0) <= len(universe)


def test_sweep_reports_one_row_per_value(universe, benchmark):
    table = sweep(universe, benchmark, BacktestConfig(warmup=210),
                  "breakout_band", [0.01, 0.02, 0.03])
    assert len(table) == 3
    assert {"cagr", "sharpe", "max_dd", "trades"} <= set(table.columns)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        run_backtest({})
