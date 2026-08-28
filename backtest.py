"""Event-driven backtester.

Replays bars forward one day at a time and calls the same ``strategy.evaluate``
the live path calls, so the test and the trader cannot drift apart.

Two rules keep the result honest:

1. **Decide on the close, fill on the next open.** A signal computed from bar
   ``t``'s close cannot be executed at that same close — you only know it once
   the bar is done. Orders queue and fill at bar ``t+1``'s open.
2. **Every fill pays.** Slippage moves the price against you and commission comes
   off the top. Breakout entries are exactly where real fills are worst, so a
   frictionless backtest of this strategy flatters it badly.

Run ``python backtest.py --demo`` for a self-contained run on synthetic data.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

import regime as regime_mod
import sizing as sizing_mod
from strategy import (
    Action,
    AssetClass,
    Position,
    StrategyParams,
    evaluate,
    prepare,
)

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    #: One-way slippage in basis points, applied adversely to every fill.
    slippage_bps: float = 5.0
    #: Commission in basis points of traded notional.
    commission_bps: float = 1.0
    #: Flat per-order fee, if your broker charges one.
    per_order_fee: float = 0.0

    def fill_price(self, reference: float, side: Action) -> float:
        slip = self.slippage_bps / 10_000.0
        return reference * (1 + slip) if side is Action.BUY else reference * (1 - slip)

    def costs(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 10_000.0 + self.per_order_fee


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    use_regime_gate: bool = True
    strategy: StrategyParams = field(default_factory=StrategyParams)
    sizing: sizing_mod.SizingParams = field(default_factory=sizing_mod.SizingParams)
    regime: regime_mod.RegimeParams = field(default_factory=regime_mod.RegimeParams)
    costs: CostModel = field(default_factory=CostModel)
    #: Bars skipped at the start so indicators are warm before trading.
    warmup: int = 200


@dataclass
class _Order:
    symbol: str
    side: Action
    quantity: float
    signal_date: pd.Timestamp
    reason: str


@dataclass
class _OpenTrade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    quantity: float
    entry_costs: float
    entry_bar: int


class BacktestResult:
    def __init__(self, equity: pd.Series, trades: pd.DataFrame, config: BacktestConfig,
                 exposure: float, rejected: int):
        self.equity = equity
        self.trades = trades
        self.config = config
        self.exposure = exposure
        self.rejected_orders = rejected
        self.metrics = _compute_metrics(equity, trades, exposure)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=" * 58,
            "BACKTEST RESULT",
            "=" * 58,
            f"Period                {m['start']:%Y-%m-%d} -> {m['end']:%Y-%m-%d} ({m['days']} bars)",
            f"Starting equity       ${self.config.initial_equity:,.0f}",
            f"Ending equity         ${m['ending_equity']:,.0f}",
            "",
            f"Total return          {m['total_return']:>8.2%}",
            f"CAGR                  {m['cagr']:>8.2%}",
            f"Annualised vol        {m['ann_vol']:>8.2%}",
            f"Sharpe (rf=0)         {m['sharpe']:>8.2f}",
            f"Max drawdown          {m['max_drawdown']:>8.2%}",
            f"Calmar                {m['calmar']:>8.2f}",
            f"Time in market        {m['exposure']:>8.2%}",
            "",
            f"Trades                {m['n_trades']:>8d}",
            f"Win rate              {m['win_rate']:>8.2%}",
            f"Avg win               {m['avg_win']:>8.2%}",
            f"Avg loss              {m['avg_loss']:>8.2%}",
            f"Profit factor         {m['profit_factor']:>8.2f}",
            f"Expectancy / trade    {m['expectancy']:>8.2%}",
            f"Avg holding (bars)    {m['avg_bars_held']:>8.1f}",
            f"Total costs paid      ${m['total_costs']:>8,.0f}",
        ]
        if self.rejected_orders:
            lines.append(f"Orders skipped (cash) {self.rejected_orders:>8d}")
        lines.append("=" * 58)
        return "\n".join(lines)


def _compute_metrics(equity: pd.Series, trades: pd.DataFrame, exposure: float) -> dict:
    out: dict = {}
    out["start"] = equity.index[0]
    out["end"] = equity.index[-1]
    out["days"] = len(equity)
    out["ending_equity"] = float(equity.iloc[-1])
    out["exposure"] = exposure

    start_val = float(equity.iloc[0])
    out["total_return"] = out["ending_equity"] / start_val - 1 if start_val > 0 else 0.0

    years = len(equity) / TRADING_DAYS
    if years > 0 and start_val > 0 and out["ending_equity"] > 0:
        out["cagr"] = (out["ending_equity"] / start_val) ** (1 / years) - 1
    else:
        out["cagr"] = 0.0

    rets = equity.pct_change().dropna()
    std = float(rets.std())
    out["ann_vol"] = std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0
    out["sharpe"] = (float(rets.mean()) / std) * math.sqrt(TRADING_DAYS) if std > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    out["max_drawdown"] = float(drawdown.min())
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else 0.0

    if trades.empty:
        out.update(n_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                   profit_factor=0.0, expectancy=0.0, avg_bars_held=0.0, total_costs=0.0)
        return out

    rp = trades["return_pct"]
    wins, losses = rp[rp > 0], rp[rp <= 0]
    gross_win = float(trades.loc[rp > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[rp <= 0, "pnl"].sum())

    out["n_trades"] = int(len(trades))
    out["win_rate"] = float(len(wins) / len(trades))
    out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    out["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf")
    out["expectancy"] = float(rp.mean())
    out["avg_bars_held"] = float(trades["bars_held"].mean())
    out["total_costs"] = float(trades["costs"].sum())
    return out


def run_backtest(
    price_data: dict[str, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    config: BacktestConfig = BacktestConfig(),
    long_term_symbols: Optional[set[str]] = None,
) -> BacktestResult:
    """Replay ``price_data`` bar by bar and return the result."""
    if not price_data:
        raise ValueError("no price data supplied")

    long_term_symbols = long_term_symbols or set()

    frames, bar_of_date, gates = {}, {}, {}
    for symbol, df in price_data.items():
        if df.empty:
            continue
        frame = prepare(df, config.strategy)
        frames[symbol] = frame
        bar_of_date[symbol] = {d: i for i, d in enumerate(frame.index)}
        if config.use_regime_gate and benchmark is not None:
            gates[symbol] = regime_mod.build_gate(benchmark, frame.index, config.regime)
        else:
            gates[symbol] = pd.Series(True, index=frame.index, dtype=bool)

    if not frames:
        raise ValueError("all supplied price frames were empty")

    all_dates = sorted(set().union(*(set(f.index) for f in frames.values())))

    cash = config.initial_equity
    positions: dict[str, Position] = {s: Position(long_term=s in long_term_symbols) for s in frames}
    open_trades: dict[str, _OpenTrade] = {}
    pending: list[_Order] = []
    closed: list[dict] = []
    equity_points: list[float] = []
    invested_days = 0
    rejected = 0

    for t, date in enumerate(all_dates):
        # --- 1. Fill orders queued on the previous bar, at today's open ---------
        still_pending: list[_Order] = []
        for order in pending:
            frame = frames[order.symbol]
            i = bar_of_date[order.symbol].get(date)
            if i is None:
                # Symbol did not trade today (holiday, listing gap). Carry the
                # order rather than silently dropping the signal.
                still_pending.append(order)
                continue

            reference = float(frame.df.iloc[i]["Open"])
            price = config.costs.fill_price(reference, order.side)

            if order.side is Action.BUY:
                quantity = order.quantity
                asset_class = AssetClass.infer(order.symbol)
                gross = quantity * price
                fee = config.costs.costs(gross)
                if gross + fee > cash:
                    affordable = max(0.0, (cash - fee) / price)
                    quantity = (round(affordable, config.sizing.crypto_precision)
                                if asset_class is AssetClass.CRYPTO else float(math.floor(affordable)))
                    gross = quantity * price
                    fee = config.costs.costs(gross)
                if quantity <= 0 or gross + fee > cash:
                    rejected += 1
                    continue

                cash -= gross + fee
                positions[order.symbol] = replace(
                    positions[order.symbol], quantity=quantity, avg_price=price
                )
                open_trades[order.symbol] = _OpenTrade(
                    order.symbol, date, price, quantity, fee, t
                )

            else:  # SELL
                held = positions[order.symbol]
                quantity = min(order.quantity, held.quantity)
                if quantity <= 0:
                    continue
                gross = quantity * price
                fee = config.costs.costs(gross)
                cash += gross - fee

                trade = open_trades.pop(order.symbol, None)
                if trade is not None:
                    total_costs = trade.entry_costs + fee
                    pnl = (price - trade.entry_price) * quantity - total_costs
                    basis = trade.entry_price * quantity
                    closed.append({
                        "symbol": order.symbol,
                        "entry_date": trade.entry_date,
                        "exit_date": date,
                        "entry_price": trade.entry_price,
                        "exit_price": price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "return_pct": pnl / basis if basis > 0 else 0.0,
                        "bars_held": t - trade.entry_bar,
                        "costs": total_costs,
                        "exit_reason": order.reason,
                    })
                positions[order.symbol] = replace(
                    positions[order.symbol], quantity=held.quantity - quantity,
                    avg_price=0.0 if held.quantity - quantity <= 0 else held.avg_price,
                )
        pending = still_pending

        # --- 2. Mark to market on today's close --------------------------------
        holdings_value = 0.0
        for symbol, pos in positions.items():
            if not pos.is_open:
                continue
            i = bar_of_date[symbol].get(date)
            if i is not None:
                holdings_value += pos.quantity * float(frames[symbol].df.iloc[i]["Close"])
            else:
                holdings_value += pos.notional(pos.avg_price)
        equity = cash + holdings_value
        equity_points.append(equity)
        if holdings_value > 0:
            invested_days += 1

        # --- 3. Decide on today's close, queue for tomorrow's open -------------
        if t >= len(all_dates) - 1:
            continue  # nothing left to fill against

        for symbol, frame in frames.items():
            i = bar_of_date[symbol].get(date)
            if i is None or i < config.warmup:
                continue
            if any(o.symbol == symbol for o in pending):
                continue  # one live order per symbol at a time

            position = positions[symbol]
            gate_ok = bool(gates[symbol].iloc[i])
            decision = evaluate(frame, i, position, regime_ok=gate_ok)

            if decision.action is Action.BUY and not position.is_open:
                ann_vol = sizing_mod.annualized_vol_from_atr(decision.atr, decision.price)
                quantity = sizing_mod.target_quantity(
                    equity, decision.price, ann_vol,
                    AssetClass.infer(symbol), config.sizing,
                )
                if quantity > 0:
                    pending.append(_Order(symbol, Action.BUY, quantity, date, "breakout"))

            elif decision.action is Action.SELL and position.is_open:
                pending.append(_Order(symbol, Action.SELL, position.quantity, date, "sma_exit"))

    # --- Liquidate anything still open at the final close -----------------------
    final_date = all_dates[-1]
    for symbol, pos in positions.items():
        if not pos.is_open:
            continue
        i = bar_of_date[symbol].get(final_date)
        if i is None:
            continue
        reference = float(frames[symbol].df.iloc[i]["Close"])
        price = config.costs.fill_price(reference, Action.SELL)
        gross = pos.quantity * price
        fee = config.costs.costs(gross)
        cash += gross - fee
        trade = open_trades.pop(symbol, None)
        if trade is not None:
            total_costs = trade.entry_costs + fee
            pnl = (price - trade.entry_price) * pos.quantity - total_costs
            basis = trade.entry_price * pos.quantity
            closed.append({
                "symbol": symbol,
                "entry_date": trade.entry_date,
                "exit_date": final_date,
                "entry_price": trade.entry_price,
                "exit_price": price,
                "quantity": pos.quantity,
                "pnl": pnl,
                "return_pct": pnl / basis if basis > 0 else 0.0,
                "bars_held": len(all_dates) - 1 - trade.entry_bar,
                "costs": total_costs,
                "exit_reason": "end_of_test",
            })
        positions[symbol] = replace(pos, quantity=0.0, avg_price=0.0)

    if equity_points:
        equity_points[-1] = cash

    equity_series = pd.Series(equity_points, index=pd.DatetimeIndex(all_dates), name="equity")
    trades_df = pd.DataFrame(closed)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    else:
        trades_df = pd.DataFrame(columns=[
            "symbol", "entry_date", "exit_date", "entry_price", "exit_price",
            "quantity", "pnl", "return_pct", "bars_held", "costs", "exit_reason",
        ])

    exposure = invested_days / len(all_dates) if all_dates else 0.0
    return BacktestResult(equity_series, trades_df, config, exposure, rejected)


def sweep(
    price_data: dict[str, pd.DataFrame],
    benchmark: Optional[pd.Series],
    base: BacktestConfig,
    param: str,
    values: list,
) -> pd.DataFrame:
    """Vary one strategy parameter and report how the result moves.

    Look for a plateau, not a peak. If a parameter works at 10 but not at 9 or 11,
    that is noise being fitted, and it will not survive contact with live data.
    """
    rows = []
    for value in values:
        cfg = replace(base, strategy=replace(base.strategy, **{param: value}))
        try:
            res = run_backtest(price_data, benchmark, cfg)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({param: value, "error": str(exc)})
            continue
        m = res.metrics
        rows.append({
            param: value,
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_drawdown"],
            "trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "expectancy": m["expectancy"],
        })
    return pd.DataFrame(rows)


def _demo(args: argparse.Namespace) -> None:
    import data as data_mod

    print("Generating synthetic data (no network required)...\n")
    prices = {
        "SYNTH_A": data_mod.synthetic_ohlcv(n=900, seed=3, annual_drift=0.14),
        "SYNTH_B": data_mod.synthetic_ohlcv(n=900, seed=17, annual_drift=0.06, annual_vol=0.45),
        "SYNTH_C": data_mod.synthetic_ohlcv(n=900, seed=29, annual_drift=0.02, annual_vol=0.22),
    }
    index = prices["SYNTH_A"].index
    bench = data_mod.synthetic_benchmark(index, seed=5, regime_flip=520)

    cfg = BacktestConfig(use_regime_gate=not args.no_regime)
    result = run_backtest(prices, bench, cfg)
    print(result.summary())

    if not result.trades.empty:
        print("\nFirst 8 trades:")
        cols = ["symbol", "entry_date", "exit_date", "return_pct", "bars_held", "exit_reason"]
        print(result.trades[cols].head(8).to_string(index=False))

    print("\nRegime gate on vs off:")
    for flag in (True, False):
        r = run_backtest(prices, bench, replace(cfg, use_regime_gate=flag))
        print(f"  gate={'on ' if flag else 'off'}  "
              f"CAGR {r.metrics['cagr']:>7.2%}  Sharpe {r.metrics['sharpe']:>5.2f}  "
              f"maxDD {r.metrics['max_drawdown']:>7.2%}  trades {r.metrics['n_trades']:>3d}")

    print("\nParameter sweep on the entry band (look for a plateau):")
    print(sweep(prices, bench, cfg, "breakout_band",
                [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]).to_string(index=False))


def _live(args: argparse.Namespace) -> None:
    import data as data_mod

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Loading {symbols} plus benchmark {args.benchmark} ({args.period})...")
    prices = data_mod.load_universe(symbols, period=args.period)
    if not prices:
        print("No data available. Check the network, or run with --demo.")
        return
    print(f"Loaded: {', '.join(f'{k} ({len(v)} bars)' for k, v in prices.items())}")

    bench_df = data_mod.load_history(args.benchmark, period=args.period)
    bench = bench_df["Close"] if not bench_df.empty else None
    if bench is None and not args.no_regime:
        print("Benchmark unavailable; running without the regime gate.")

    cfg = BacktestConfig(
        initial_equity=args.equity,
        use_regime_gate=(bench is not None and not args.no_regime),
    )
    result = run_backtest(prices, bench, cfg)
    print()
    print(result.summary())
    if not result.trades.empty and args.show_trades:
        print("\nTrades:")
        print(result.trades.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the breakout strategy.")
    parser.add_argument("--demo", action="store_true",
                        help="run on deterministic synthetic data, no network needed")
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA", help="comma-separated tickers")
    parser.add_argument("--benchmark", default="^GSPC", help="regime benchmark symbol")
    parser.add_argument("--period", default="5y", help="history window, e.g. 5y")
    parser.add_argument("--equity", type=float, default=100_000.0, help="starting equity")
    parser.add_argument("--no-regime", action="store_true", help="disable the regime gate")
    parser.add_argument("--show-trades", action="store_true", help="print the full trade list")
    args = parser.parse_args()

    if args.demo:
        _demo(args)
    else:
        _live(args)


if __name__ == "__main__":
    main()
