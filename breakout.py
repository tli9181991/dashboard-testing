"""Live monitoring path.

This is a thin shell around ``strategy.evaluate`` — the same function the
backtester calls. All trading logic lives in ``strategy.py``; if you change a rule,
change it there and both paths move together. A live path with its own copy of the
rules is a backtest that silently stops describing the system you are running.

Responsibilities kept here: fetching bars, expiring the cache, dropping the
still-forming bar, applying the regime gate, and reporting a vol-targeted size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

import regime as regime_mod
import sizing as sizing_mod
from config import CACHE_DIR
from strategy import (
    Action,
    AssetClass,
    Decision,
    Position,
    StrategyParams,
    drop_forming_bar,
    evaluate,
    prepare,
)

#: Cached bars older than this are refetched. The original code cached price
#: history forever, so a dashboard that auto-refreshed every 30 seconds could sit
#: on days-old prices while looking live.
CACHE_TTL_SECONDS = 60 * 30


@dataclass
class MarketView:
    """Everything the dashboard needs for one symbol."""

    symbol: str
    df: pd.DataFrame
    decision: Decision
    position: Position
    asset_class: AssetClass
    unrealized_pnl: float
    ann_vol: float
    target_quantity: float
    regime_ok: bool

    @property
    def price(self) -> float:
        return self.decision.price

    @property
    def signal(self) -> str:
        return self.decision.action.value

    @property
    def logs(self) -> tuple[str, ...]:
        return self.decision.logs

    @property
    def target_notional(self) -> float:
        return self.target_quantity * self.price


class SwingBreakoutMonitor:
    def __init__(
        self,
        symbol: str,
        position: Position = Position(),
        equity: float = 100_000.0,
        params: StrategyParams = StrategyParams(),
        sizing_params: sizing_mod.SizingParams = sizing_mod.SizingParams(),
        period: str = "2y",
    ):
        self.symbol = symbol.upper()
        self.position = position
        self.equity = equity
        self.params = params
        self.sizing_params = sizing_params
        self.period = period
        self.asset_class = AssetClass.infer(self.symbol)
        self.cache_file = CACHE_DIR / f"{self.symbol.replace('-', '_')}_price_history.csv"

    @property
    def is_crypto(self) -> bool:
        return self.asset_class is AssetClass.CRYPTO

    def _cache_is_fresh(self) -> bool:
        if not self.cache_file.exists():
            return False
        return (time.time() - self.cache_file.stat().st_mtime) < CACHE_TTL_SECONDS

    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self._cache_is_fresh():
            try:
                cached = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                if not cached.empty:
                    return cached
            except Exception:
                pass

        import yfinance as yf

        df = yf.download(self.symbol, period=self.period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        if not df.empty:
            df.to_csv(self.cache_file)
        return df

    def evaluate_market(
        self,
        force_refresh: bool = False,
        benchmark_close: Optional[pd.Series] = None,
        now: Optional[pd.Timestamp] = None,
    ) -> MarketView:
        raw = self.fetch_data(force_refresh=force_refresh)
        if raw.empty:
            raise ValueError(f"no price history available for {self.symbol}")

        # Decide on completed bars only. The exit rule is defined on the close;
        # evaluating it against a bar that is still printing makes the signal
        # fire and unfire intraday.
        closed = drop_forming_bar(raw, self.asset_class, now=now)
        if closed.empty:
            raise ValueError(f"{self.symbol}: no completed bars available")

        frame = prepare(closed, self.params)
        last = len(frame) - 1

        if benchmark_close is not None:
            gate = regime_mod.build_gate(benchmark_close, frame.index)
            regime_ok = bool(gate.iloc[last])
        else:
            regime_ok = True

        decision = evaluate(frame, last, self.position, regime_ok=regime_ok)
        ann_vol = sizing_mod.annualized_vol_from_atr(decision.atr, decision.price)
        target = sizing_mod.target_quantity(
            self.equity, decision.price, ann_vol, self.asset_class, self.sizing_params
        )

        return MarketView(
            symbol=self.symbol,
            df=frame.df,
            decision=decision,
            position=self.position,
            asset_class=self.asset_class,
            unrealized_pnl=self.position.unrealized_pnl(decision.price),
            ann_vol=ann_vol,
            target_quantity=target,
            regime_ok=regime_ok,
        )


# Re-exported so existing imports keep resolving.
__all__ = ["SwingBreakoutMonitor", "MarketView", "Action", "Position", "CACHE_TTL_SECONDS"]
