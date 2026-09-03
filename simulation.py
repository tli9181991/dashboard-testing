"""Monte Carlo layer over a backtest result.

A single historical replay produces one number. That number is the joint outcome
of the strategy, the sequence the trades happened to arrive in, and the particular
stretch of market history it ran through — and it cannot tell you how much of the
result was any of the three. This module separates them.

Three questions, three tools:

**How much of the result was sequence luck?** ``bootstrap_paths`` resamples the
realised trade returns with replacement and rebuilds the equity curve thousands of
times. The spread of final outcomes is the answer. It assumes trades are
independent — a block bootstrap (``block_size > 1``) keeps short runs intact if you
think winners and losers cluster.

**Did the signal beat simply being exposed?** ``random_entry_benchmark`` takes the
same number of trades with the same holding periods, but enters on random dates in
the same names. If the strategy's average trade does not clear that distribution,
the entry rule is decoration and the returns came from exposure alone. This is the
sharpest of the three.

**Did the trading beat not trading?** ``buy_and_hold`` is the honest floor.

A caution that applies to all of them, and especially when the universe came from a
screen: stocks selected today for having trended are not a fair sample of the
stocks you could have selected back then. Absolute returns from such a run are
inflated and should not be read as an edge. The *comparisons* survive, because the
strategy, buy-and-hold and the random-entry benchmark all inherit the same bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class SimulationParams:
    n_paths: int = 2000
    seed: int = 7
    #: 1 resamples individual trades; larger keeps runs of that length together,
    #: preserving any streakiness in the original sequence.
    block_size: int = 1


@dataclass
class PathResult:
    """Equity multiples for every simulated path, shape (n_paths, n_trades + 1)."""

    paths: np.ndarray
    params: SimulationParams

    @property
    def finals(self) -> np.ndarray:
        return self.paths[:, -1]

    def percentile_band(self, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
        return (np.percentile(self.paths, lower, axis=0),
                np.percentile(self.paths, upper, axis=0))

    @property
    def median_path(self) -> np.ndarray:
        return np.percentile(self.paths, 50, axis=0)

    def summary(self) -> dict:
        finals = self.finals
        return {
            "n_paths": int(self.paths.shape[0]),
            "n_trades": int(self.paths.shape[1] - 1),
            "median_return": float(np.median(finals) - 1),
            "p05_return": float(np.percentile(finals, 5) - 1),
            "p95_return": float(np.percentile(finals, 95) - 1),
            "prob_profit": float((finals > 1).mean()),
            "median_max_drawdown": float(np.median(max_drawdowns(self.paths))),
            "worst_max_drawdown": float(max_drawdowns(self.paths).min()),
        }


def max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Worst peak-to-trough decline on each path, as a negative fraction."""
    running_max = np.maximum.accumulate(paths, axis=1)
    return (paths / running_max - 1).min(axis=1)


def bootstrap_paths(
    trade_returns,
    params: SimulationParams = SimulationParams(),
    n_trades: Optional[int] = None,
) -> Optional[PathResult]:
    """Resample the trade sequence to see the range of outcomes it could have given.

    Returns None when there are too few trades for the answer to mean anything.
    """
    returns = np.asarray(pd.Series(trade_returns).dropna(), dtype=float)
    if returns.size < 2:
        return None

    n_trades = int(n_trades or returns.size)
    if n_trades < 1:
        return None

    rng = np.random.default_rng(params.seed)
    block = max(1, int(params.block_size))

    if block == 1:
        draws = rng.choice(returns, size=(params.n_paths, n_trades), replace=True)
    else:
        # Block bootstrap: start indices are random, each block runs forward and
        # wraps, so runs of consecutive trades stay together.
        n_blocks = math.ceil(n_trades / block)
        starts = rng.integers(0, returns.size, size=(params.n_paths, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]) % returns.size
        draws = returns[idx].reshape(params.n_paths, -1)[:, :n_trades]

    growth = np.cumprod(1.0 + draws, axis=1)
    paths = np.hstack([np.ones((params.n_paths, 1)), growth])
    return PathResult(paths, params)


def buy_and_hold(price_data: dict[str, pd.DataFrame]) -> dict:
    """Equal-weight buy and hold across the same names, over the same window."""
    if not price_data:
        return {}

    per_symbol, curves = {}, []
    for symbol, frame in price_data.items():
        close = frame["Close"].dropna()
        if len(close) < 2:
            continue
        normalised = close / float(close.iloc[0])
        per_symbol[symbol] = float(normalised.iloc[-1] - 1)
        curves.append(normalised)

    if not curves:
        return {}

    combined = pd.concat(curves, axis=1).ffill().dropna(how="all")
    portfolio = combined.mean(axis=1)
    total = float(portfolio.iloc[-1] - 1)
    years = len(portfolio) / TRADING_DAYS

    running_max = portfolio.cummax()
    return {
        "per_symbol": per_symbol,
        "curve": portfolio,
        "total_return": total,
        "cagr": (portfolio.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0,
        "max_drawdown": float((portfolio / running_max - 1).min()),
    }


def random_entry_benchmark(
    price_data: dict[str, pd.DataFrame],
    trades: pd.DataFrame,
    params: SimulationParams = SimulationParams(),
    cost_bps: float = 12.0,
) -> Optional[dict]:
    """Does the entry rule beat entering at random with the same exposure?

    Each simulated path replays the same number of trades with the same holding
    periods, but picks entry dates uniformly at random in the same names. The
    strategy's realised average trade is then placed within that distribution.

    A percentile near 50 means the signal added nothing over being in the market
    for the same amount of time; the returns came from exposure, not selection.
    """
    if trades is None or trades.empty or not price_data:
        return None

    holds = trades["bars_held"].astype(int).clip(lower=1).to_numpy()
    if holds.size < 2:
        return None

    usable = {s: f["Close"].dropna().to_numpy(dtype=float)
              for s, f in price_data.items() if len(f["Close"].dropna()) > holds.max() + 2}
    if not usable:
        return None

    symbols = list(usable)
    rng = np.random.default_rng(params.seed + 1)
    cost = cost_bps / 10_000.0

    means = np.empty(params.n_paths, dtype=float)
    for p in range(params.n_paths):
        picks = rng.integers(0, len(symbols), size=holds.size)
        drawn = np.empty(holds.size, dtype=float)
        for k, (sym_idx, hold) in enumerate(zip(picks, holds)):
            closes = usable[symbols[sym_idx]]
            last_start = closes.size - hold - 1
            start = int(rng.integers(0, max(last_start, 1)))
            drawn[k] = closes[start + hold] / closes[start] - 1.0 - cost
        means[p] = drawn.mean()

    actual = float(trades["return_pct"].mean())
    return {
        "actual_mean_trade": actual,
        "random_mean_trade": float(means.mean()),
        "percentile": float((means < actual).mean() * 100),
        "distribution": means,
        "n_paths": params.n_paths,
        "n_trades": int(holds.size),
    }


def summarise(
    result,
    price_data: dict[str, pd.DataFrame],
    params: SimulationParams = SimulationParams(),
) -> dict:
    """Run all three comparisons against a BacktestResult."""
    out: dict = {"metrics": dict(result.metrics)}

    boot = bootstrap_paths(result.trades["return_pct"], params) if not result.trades.empty else None
    out["bootstrap"] = boot
    out["bootstrap_summary"] = boot.summary() if boot is not None else None
    out["buy_and_hold"] = buy_and_hold(price_data)
    out["random_entry"] = random_entry_benchmark(price_data, result.trades, params)
    return out
