"""Triple-barrier backtester for the Swing Universe Funnel setups.

``backtest.py`` drives the breakout rule in ``strategy.py``, whose exit is a moving
average. The swing setups are a different shape: each produces a complete bracket
— an entry order, a stop, and two targets — so it needs an engine that places
resting orders and manages three barriers.

What is modelled, and why each detail matters
---------------------------------------------

**Causality.** At each bar the setup functions are called with history truncated to
that bar, exactly as they are called live. They are causal by construction —
``swing_points`` lags its fractals and ``prior_period_levels`` uses only completed
periods — and ``tests/test_swing_backtest.py`` asserts the replay never revises a
decision once made.

**Resting orders, not instant fills.** §04A's entry sits above the trigger bar's
high, so it is a stop-buy that fills only if price trades up through it. §04B's
entry is the proximal edge of a gap below price, so it is a limit-buy that fills
only if price trades back down into it. Either order expires unfilled after a few
sessions; a setup that never triggered is not a trade, and counting it as one is
the most common way this kind of backtest lies.

**Gap fills.** A stop-buy that gaps through its trigger fills at the open, worse
than the order price. A limit-buy that gaps below fills at the open, better. Both
are modelled; assuming you always get your price overstates the result.

**The intrabar unknown.** When a daily bar's range covers both the stop and a
target, daily OHLC cannot say which came first. Rather than pick silently,
``intrabar`` runs "stop" (pessimistic) or "target" (optimistic), and
``ambiguity_bound`` runs both so the gap between them is visible. A strategy whose
result depends heavily on that gap has not been measured, it has been guessed.

**Partial exits.** §04A's own comment says TP1 is where you take a partial and move
the stop, not where the trade pays. So TP1 sells a fraction and moves the stop to
breakeven; the rest runs to TP2 or the time stop.

MAE and MFE are recorded in R for every trade, which is what tells you whether the
stops are placed where the trades actually need them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

import swing_screener as swing

TRADING_DAYS = 252
Intrabar = Literal["stop", "target"]


@dataclass(frozen=True)
class SwingBacktestConfig:
    initial_equity: float = 100_000.0
    #: Sessions a resting entry order stays live before it is cancelled.
    order_ttl: int = 3
    #: Sessions a filled position is held before the time stop closes it.
    max_hold: int = 30
    #: Fraction sold at TP1; the stop then moves to breakeven on the remainder.
    tp1_fraction: float = 0.5
    #: Which barrier wins when one bar covers both.
    intrabar: Intrabar = "stop"
    slippage_bps: float = 8.0
    commission_bps: float = 1.0
    #: Bars skipped so the setups have history to work with.
    warmup: int = 260
    #: Evaluate setups every N bars. 1 is faithful; higher is faster.
    step: int = 1
    #: Bars of history handed to each setup call. Must comfortably exceed the
    #: 252 the trend template needs.
    setup_window: int = 420
    use_regime: bool = True
    max_concurrent: Optional[int] = None
    variants: tuple[str, ...] = ("A/momentum", "B/ict")


@dataclass
class _Pending:
    symbol: str
    variant: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    shares: int
    placed_bar: int
    note: str

    @property
    def is_stop_buy(self) -> bool:
        # §04A triggers above the market, §04B rests below it.
        return self.variant.startswith("A")


@dataclass
class _Open:
    symbol: str
    variant: str
    entry_price: float
    stop: float
    tp1: float
    tp2: float
    shares: int
    remaining: int
    entry_bar: int
    entry_date: pd.Timestamp
    risk_per_share: float
    entry_costs: float
    realised: float = 0.0
    took_partial: bool = False
    mae_r: float = 0.0
    mfe_r: float = 0.0
    note: str = ""


TRADE_COLUMNS = [
    "symbol", "variant", "entry_date", "exit_date", "entry_price", "exit_price",
    "shares", "pnl", "return_pct", "r_multiple", "bars_held", "costs",
    "mae_r", "mfe_r", "took_partial", "exit_reason", "note",
]


class SwingBacktestResult:
    def __init__(self, equity: pd.Series, trades: pd.DataFrame,
                 config: SwingBacktestConfig, stats: dict):
        self.equity = equity
        self.trades = trades
        self.config = config
        self.stats = stats
        self.metrics = _metrics(equity, trades, config, stats)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=" * 60,
            f"SWING BACKTEST  ({self.config.intrabar}-first on ambiguous bars)",
            "=" * 60,
            f"Period              {m['start']:%Y-%m-%d} -> {m['end']:%Y-%m-%d}",
            f"Ending equity       ${m['ending_equity']:,.0f}",
            "",
            f"Total return        {m['total_return']:>8.2%}",
            f"CAGR                {m['cagr']:>8.2%}",
            f"Sharpe (rf=0)       {m['sharpe']:>8.2f}",
            f"Max drawdown        {m['max_drawdown']:>8.2%}",
            "",
            f"Setups seen         {m['setups_seen']:>8d}",
            f"Orders placed       {m['orders_placed']:>8d}",
            f"Orders filled       {m['n_trades']:>8d}  ({m['fill_rate']:.0%})",
            f"Expired unfilled    {m['orders_expired']:>8d}",
            "",
            f"Win rate            {m['win_rate']:>8.2%}",
            f"Avg R               {m['avg_r']:>8.2f}",
            f"Expectancy (R)      {m['expectancy_r']:>8.2f}",
            f"Profit factor       {m['profit_factor']:>8.2f}",
            f"Median MAE (R)      {m['median_mae_r']:>8.2f}",
            f"Median MFE (R)      {m['median_mfe_r']:>8.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _metrics(equity: pd.Series, trades: pd.DataFrame,
             config: SwingBacktestConfig, stats: dict) -> dict:
    out: dict = dict(stats)
    out["start"] = equity.index[0]
    out["end"] = equity.index[-1]
    out["ending_equity"] = float(equity.iloc[-1])

    start = float(equity.iloc[0])
    out["total_return"] = out["ending_equity"] / start - 1 if start > 0 else 0.0

    years = len(equity) / TRADING_DAYS
    out["cagr"] = ((out["ending_equity"] / start) ** (1 / years) - 1
                   if years > 0 and start > 0 and out["ending_equity"] > 0 else 0.0)

    rets = equity.pct_change().dropna()
    sd = float(rets.std())
    out["ann_vol"] = sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    out["sharpe"] = (float(rets.mean()) / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    dd = equity / equity.cummax() - 1
    out["max_drawdown"] = float(dd.min())

    placed = out.get("orders_placed", 0)
    out["n_trades"] = int(len(trades))
    out["fill_rate"] = (len(trades) / placed) if placed else 0.0

    if trades.empty:
        out.update(win_rate=0.0, avg_r=0.0, expectancy_r=0.0, profit_factor=0.0,
                   median_mae_r=0.0, median_mfe_r=0.0, avg_bars_held=0.0,
                   total_costs=0.0)
        return out

    r = trades["r_multiple"]
    wins, losses = trades[r > 0], trades[r <= 0]
    gross_win = float(wins["pnl"].sum())
    gross_loss = float(-losses["pnl"].sum())

    out["win_rate"] = float(len(wins) / len(trades))
    out["avg_r"] = float(r.mean())
    out["expectancy_r"] = float(r.mean())
    out["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf")
    out["median_mae_r"] = float(trades["mae_r"].median())
    out["median_mfe_r"] = float(trades["mfe_r"].median())
    out["avg_bars_held"] = float(trades["bars_held"].mean())
    out["total_costs"] = float(trades["costs"].sum())
    return out


def _breadth_series(price_data: dict[str, pd.DataFrame], index: pd.Index) -> pd.Series:
    """Fraction of the universe above its own 50-day average, per date.

    Rolling means look backwards only, so the value on any date uses that date
    and earlier — the same causality the rest of the engine relies on.
    """
    flags = []
    for frame in price_data.values():
        close = frame["Close"]
        if len(close) < 50:
            continue
        flags.append((close > close.rolling(50).mean()).reindex(index).astype(float))
    if not flags:
        return pd.Series(0.5, index=index)
    return pd.concat(flags, axis=1).mean(axis=1).fillna(0.0)


def _fill_price(order: _Pending, bar: pd.Series, slip: float) -> Optional[float]:
    """Did the resting order trade, and at what price?"""
    open_, high, low = float(bar["Open"]), float(bar["High"]), float(bar["Low"])

    if order.is_stop_buy:
        if high < order.entry:
            return None
        # A gap through the trigger fills at the open, worse than the order.
        raw = max(order.entry, open_)
        return raw * (1 + slip)

    if low > order.entry:
        return None
    # A gap below a limit fills at the open, better than the order.
    raw = min(order.entry, open_)
    return raw * (1 + slip)


def run_swing_backtest(
    price_data: dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    cfg: Optional[dict] = None,
    config: SwingBacktestConfig = SwingBacktestConfig(),
    progress: Optional[Callable[[float], None]] = None,
) -> SwingBacktestResult:
    """Replay the swing setups bar by bar and manage their brackets."""
    if not price_data:
        raise ValueError("no price data supplied")

    cfg = dict(cfg or swing.CFG)
    frames = {s: f for s, f in price_data.items() if len(f) > config.warmup + 5}
    if not frames:
        raise ValueError("no symbol has enough history for the configured warmup")

    dates = sorted(set().union(*(set(f.index) for f in frames.values())))
    bar_of = {s: {d: i for i, d in enumerate(f.index)} for s, f in frames.items()}
    breadth = _breadth_series(frames, pd.DatetimeIndex(dates))
    # Prior-period liquidity pools depend only on completed periods, so they are
    # computed once per symbol here and sliced per bar rather than recomputed on
    # every setup call. Equivalence is asserted in the tests.
    level_cache = {s: swing.prior_period_levels(f) for s, f in frames.items()}
    spy_close = spy["Close"] if "Close" in spy else spy
    spy_aligned = spy_close.reindex(pd.DatetimeIndex(dates)).ffill()

    slip = config.slippage_bps / 10_000.0
    fee_rate = config.commission_bps / 10_000.0

    cash = config.initial_equity
    pending: list[_Pending] = []
    positions: dict[str, _Open] = {}
    closed: list[dict] = []
    equity_points: list[float] = []
    stats = {"setups_seen": 0, "orders_placed": 0, "orders_expired": 0,
             "orders_skipped_cash": 0, "regime_blocked": 0}

    def fee(notional: float) -> float:
        return abs(notional) * fee_rate

    for t, date in enumerate(dates):
        # ── 1. resting orders ────────────────────────────────────────────────
        survivors: list[_Pending] = []
        for order in pending:
            i = bar_of[order.symbol].get(date)
            if i is None:
                survivors.append(order)
                continue
            if t - order.placed_bar > config.order_ttl:
                stats["orders_expired"] += 1
                continue

            bar = frames[order.symbol].iloc[i]
            price = _fill_price(order, bar, slip)
            if price is None:
                survivors.append(order)
                continue

            shares = order.shares
            cost = shares * price
            charge = fee(cost)
            if cost + charge > cash:
                shares = int(max(0.0, (cash - charge) / price))
                cost = shares * price
                charge = fee(cost)
            if shares < 1 or cost + charge > cash:
                stats["orders_skipped_cash"] += 1
                continue

            cash -= cost + charge
            risk = price - order.stop
            if risk <= 0:
                cash += cost - charge          # unwind: the gap invalidated the bracket
                continue

            positions[order.symbol] = _Open(
                symbol=order.symbol, variant=order.variant, entry_price=price,
                stop=order.stop, tp1=order.tp1, tp2=order.tp2, shares=shares,
                remaining=shares, entry_bar=t, entry_date=date,
                risk_per_share=risk, entry_costs=charge, note=order.note,
            )
        pending = survivors

        # ── 2. manage open positions ─────────────────────────────────────────
        for symbol in list(positions):
            pos = positions[symbol]
            i = bar_of[symbol].get(date)
            if i is None or i <= bar_of[symbol].get(pos.entry_date, -1):
                continue

            bar = frames[symbol].iloc[i]
            open_, high, low, close = (float(bar[c]) for c in ("Open", "High", "Low", "Close"))

            pos.mae_r = min(pos.mae_r, (low - pos.entry_price) / pos.risk_per_share)
            pos.mfe_r = max(pos.mfe_r, (high - pos.entry_price) / pos.risk_per_share)

            target = pos.tp2 if pos.took_partial else pos.tp1
            hit_stop, hit_target = low <= pos.stop, high >= target

            if hit_stop and hit_target:
                # One daily bar covered both. OHLC cannot resolve the order.
                hit_target = config.intrabar == "target"
                hit_stop = not hit_target

            def close_out(price: float, shares: int, reason: str) -> None:
                nonlocal cash
                gross = shares * price
                charge = fee(gross)
                cash += gross - charge
                total_costs = pos.entry_costs + charge + 0.0
                pnl = pos.realised + (price - pos.entry_price) * shares - total_costs
                basis = pos.entry_price * pos.shares
                closed.append({
                    "symbol": symbol, "variant": pos.variant,
                    "entry_date": pos.entry_date, "exit_date": date,
                    "entry_price": pos.entry_price, "exit_price": price,
                    "shares": pos.shares, "pnl": pnl,
                    "return_pct": pnl / basis if basis > 0 else 0.0,
                    "r_multiple": pnl / (pos.risk_per_share * pos.shares)
                    if pos.risk_per_share > 0 and pos.shares else 0.0,
                    "bars_held": t - pos.entry_bar, "costs": total_costs,
                    "mae_r": pos.mae_r, "mfe_r": pos.mfe_r,
                    "took_partial": pos.took_partial, "exit_reason": reason,
                    "note": pos.note,
                })
                positions.pop(symbol, None)

            if hit_stop:
                price = min(pos.stop, open_) * (1 - slip)      # a gap through fills worse
                close_out(price, pos.remaining, "stop" if not pos.took_partial else "breakeven")
                continue

            if hit_target:
                price = max(target, open_) * (1 - slip)        # a gap above fills better
                if not pos.took_partial and config.tp1_fraction < 1.0:
                    part = int(pos.remaining * config.tp1_fraction)
                    if part >= 1:
                        gross = part * price
                        charge = fee(gross)
                        cash += gross - charge
                        pos.realised += (price - pos.entry_price) * part - charge
                        pos.remaining -= part
                        pos.took_partial = True
                        pos.stop = pos.entry_price             # breakeven on the rest
                        continue
                close_out(price, pos.remaining, "tp2" if pos.took_partial else "tp1")
                continue

            if t - pos.entry_bar >= config.max_hold:
                close_out(close * (1 - slip), pos.remaining, "time_stop")

        # ── 3. mark to market ────────────────────────────────────────────────
        held = 0.0
        for symbol, pos in positions.items():
            i = bar_of[symbol].get(date)
            price = float(frames[symbol].iloc[i]["Close"]) if i is not None else pos.entry_price
            held += pos.remaining * price
        equity = cash + held
        equity_points.append(equity)

        # ── 4. look for new setups on this close ─────────────────────────────
        if t < config.warmup or t >= len(dates) - 1 or (t - config.warmup) % config.step:
            if progress and t % 50 == 0:
                progress(t / len(dates))
            continue

        state = "risk_on"
        if config.use_regime:
            spy_slice = spy_aligned.iloc[:t + 1].dropna()
            if len(spy_slice) > 200:
                state = swing.regime_state(
                    pd.DataFrame({"Close": spy_slice}), float(breadth.iloc[t]), None, cfg
                )
        allowed = config.max_concurrent or swing.positions_allowed(state, cfg)
        if len(positions) + len(pending) >= allowed:
            stats["regime_blocked"] += 1
            if progress and t % 50 == 0:
                progress(t / len(dates))
            continue

        for symbol, frame in frames.items():
            if symbol in positions or any(o.symbol == symbol for o in pending):
                continue
            i = bar_of[symbol].get(date)
            if i is None or i < config.warmup:
                continue

            # The setups are handed history truncated to this bar, exactly as
            # they are called live. The front is also capped: nothing in either
            # setup reaches back beyond ~260 bars (trend_template needs 252), so
            # this bounds the per-call cost without changing a decision. The
            # equivalence is asserted in tests/test_swing_backtest.py.
            lo = max(0, i + 1 - config.setup_window)
            window = frame.iloc[lo:i + 1]
            for setup in (swing.momentum_setup(window, cfg),
                          swing.ict_setup(window, cfg, level_cache[symbol])):
                if not setup or setup["variant"] not in config.variants:
                    continue
                stats["setups_seen"] += 1
                shares = swing.position_size(setup["entry"], setup["stop"], window, state, cfg)
                if shares < 1:
                    continue
                pending.append(_Pending(
                    symbol=symbol, variant=setup["variant"], entry=float(setup["entry"]),
                    stop=float(setup["stop"]), tp1=float(setup["tp1"]),
                    tp2=float(setup["tp2"]), shares=shares, placed_bar=t,
                    note=str(setup["note"]),
                ))
                stats["orders_placed"] += 1
                break

        if progress and t % 50 == 0:
            progress(t / len(dates))

    # ── liquidate whatever is still open ────────────────────────────────────
    final = dates[-1]
    for symbol in list(positions):
        pos = positions[symbol]
        i = bar_of[symbol].get(final)
        if i is None:
            continue
        price = float(frames[symbol].iloc[i]["Close"]) * (1 - slip)
        gross = pos.remaining * price
        charge = fee(gross)
        cash += gross - charge
        total_costs = pos.entry_costs + charge
        pnl = pos.realised + (price - pos.entry_price) * pos.remaining - total_costs
        basis = pos.entry_price * pos.shares
        closed.append({
            "symbol": symbol, "variant": pos.variant, "entry_date": pos.entry_date,
            "exit_date": final, "entry_price": pos.entry_price, "exit_price": price,
            "shares": pos.shares, "pnl": pnl,
            "return_pct": pnl / basis if basis > 0 else 0.0,
            "r_multiple": pnl / (pos.risk_per_share * pos.shares)
            if pos.risk_per_share > 0 and pos.shares else 0.0,
            "bars_held": len(dates) - 1 - pos.entry_bar, "costs": total_costs,
            "mae_r": pos.mae_r, "mfe_r": pos.mfe_r, "took_partial": pos.took_partial,
            "exit_reason": "end_of_test", "note": pos.note,
        })
        positions.pop(symbol, None)

    if equity_points:
        equity_points[-1] = cash

    equity = pd.Series(equity_points, index=pd.DatetimeIndex(dates), name="equity")
    trades = pd.DataFrame(closed, columns=TRADE_COLUMNS)
    if not trades.empty:
        trades = trades.sort_values("entry_date").reset_index(drop=True)
    return SwingBacktestResult(equity, trades, config, stats)


def ambiguity_bound(
    price_data: dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    cfg: Optional[dict] = None,
    config: SwingBacktestConfig = SwingBacktestConfig(),
) -> dict:
    """Run both resolutions of the intrabar unknown and report the gap.

    Daily bars cannot say whether the stop or the target came first when one bar
    covers both. The honest answer is a range: a strategy whose return swings
    wildly between these two runs has not been measured on daily data, and needs
    intraday bars before anyone trades it.
    """
    pessimistic = run_swing_backtest(price_data, spy, cfg, replace(config, intrabar="stop"))
    optimistic = run_swing_backtest(price_data, spy, cfg, replace(config, intrabar="target"))
    lo = pessimistic.metrics["total_return"]
    hi = optimistic.metrics["total_return"]
    return {
        "pessimistic": pessimistic,
        "optimistic": optimistic,
        "return_low": lo,
        "return_high": hi,
        "spread": hi - lo,
        "ambiguous_share": _ambiguous_share(pessimistic, optimistic),
    }


def _ambiguous_share(pessimistic, optimistic) -> float:
    """Fraction of trades whose outcome flipped with the tie-break."""
    a, b = pessimistic.trades, optimistic.trades
    if a.empty or b.empty:
        return 0.0
    key = ["symbol", "entry_date"]
    merged = a[key + ["exit_reason"]].merge(b[key + ["exit_reason"]], on=key,
                                            how="inner", suffixes=("_p", "_o"))
    if merged.empty:
        return 0.0
    return float((merged["exit_reason_p"] != merged["exit_reason_o"]).mean())
