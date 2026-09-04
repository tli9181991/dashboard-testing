"""Causal breakout signal generation.

This module is the single source of truth for the strategy. Both the live path
(``breakout.SwingBreakoutMonitor``) and ``backtest.py`` call ``evaluate`` here, so
a backtest cannot silently diverge from what runs against the broker.

Causality contract
------------------
Every value used to make a decision at bar ``i`` is derived only from bars
``0..i``. Concretely:

* Indicators use pandas rolling/ewm windows, which are backward-looking.
* Swing levels are detected with a *centred* window of ``2*swing_window+1`` bars,
  so a level at bar ``j`` is only knowable once bar ``j + swing_window`` exists.
  Each level records that ``confirmed_at`` index and is filtered out of any
  decision taken before it.
* The merge tolerance uses the rolling ATR *as of bar i*, not a full-sample mean.

``tests/test_causality.py`` asserts this empirically: appending future bars must
never change a decision already taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd

OHLCV = ("Open", "High", "Low", "Close", "Volume")


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"

    @staticmethod
    def infer(symbol: str) -> "AssetClass":
        return AssetClass.CRYPTO if "-USD" in symbol.upper() else AssetClass.EQUITY


@dataclass(frozen=True)
class StrategyParams:
    """Strategy knobs. Kept in one place so the backtester can sweep them."""

    sma_exit: int = 10
    atr_window: int = 14
    swing_window: int = 3
    #: A swing must stand this far above the surrounding trough to count.
    min_prominence_frac: float = 0.015
    #: Levels older than this many bars are considered stale and ignored.
    level_lookback: int = 252
    #: Entry fires while price sits within this fraction above a broken level.
    breakout_band: float = 0.02
    #: Multiple of ATR used to merge nearby swing levels into one.
    level_merge_atr_mult: float = 1.0
    ema_spans: tuple[int, ...] = (5, 10, 20, 200)


@dataclass(frozen=True)
class Position:
    """Explicit position state.

    ``quantity`` is always a *unit count* (shares or coins) — never a dollar
    amount. The original code overloaded one field to mean dollars for crypto and
    shares for equities, which made PnL and position caps asset-class dependent.
    """

    quantity: float = 0.0
    avg_price: float = 0.0
    long_term: bool = False

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def notional(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        if not self.is_open or self.avg_price <= 0:
            return 0.0
        return (price - self.avg_price) * self.quantity


@dataclass(frozen=True)
class Level:
    confirmed_at: int
    bar: int
    price: float
    kind: str


@dataclass(frozen=True)
class Decision:
    action: Action
    price: float
    sma_exit: float
    atr: float
    broken_resistance: Optional[float] = None
    next_resistance: Optional[float] = None
    regime_ok: bool = True
    logs: tuple[str, ...] = field(default_factory=tuple)


class StrategyFrame:
    """Indicator-annotated price history plus causally-tagged swing levels."""

    def __init__(self, df: pd.DataFrame, levels: Sequence[Level], params: StrategyParams):
        self.df = df
        self.levels = tuple(levels)
        self.params = params

    def __len__(self) -> int:
        return len(self.df)

    @property
    def index(self) -> pd.Index:
        return self.df.index

    def levels_asof(self, i: int) -> list[Level]:
        """Levels a decision at bar ``i`` is allowed to see."""
        floor = i - self.params.level_lookback
        return [lv for lv in self.levels if lv.confirmed_at <= i and lv.bar >= floor]


def average_true_range(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift()
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(n, min_periods=1).mean()


def add_indicators(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> pd.DataFrame:
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"price frame missing columns: {missing}")

    out = df.copy()
    out[f"SMA_{params.sma_exit}"] = out["Close"].rolling(params.sma_exit).mean()
    for span in params.ema_spans:
        out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean()
    out["ATR"] = average_true_range(out, params.atr_window)
    return out


def detect_levels(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> list[Level]:
    """Find swing highs/lows using a strictly local, centred window.

    A bar ``j`` is a swing high when its high is the maximum of
    ``[j-w, j+w]`` and it stands ``min_prominence_frac`` above the lowest low in
    that same window. Both tests depend only on bars within the window, so the
    level becomes knowable exactly at bar ``j + w`` — recorded as ``confirmed_at``.

    This replaces ``scipy.signal.find_peaks``, whose ``prominence`` is computed
    against the whole series and therefore leaks information from bars that had
    not printed yet.
    """
    w = params.swing_window
    span = 2 * w + 1
    if len(df) < span:
        return []

    high, low = df["High"], df["Low"]
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    # Centred windows: at bar j these span exactly [j-w, j+w].
    window_high = high.rolling(span, center=True).max().to_numpy(dtype=float)
    window_low = low.rolling(span, center=True).min().to_numpy(dtype=float)

    levels: list[Level] = []
    for j in range(w, len(df) - w):
        if np.isnan(window_high[j]) or np.isnan(window_low[j]):
            continue
        depth = window_high[j] - window_low[j]
        if highs[j] >= window_high[j] and highs[j] > 0 and depth / highs[j] >= params.min_prominence_frac:
            levels.append(Level(j + w, j, float(highs[j]), "resistance"))
        if lows[j] <= window_low[j] and lows[j] > 0 and depth / lows[j] >= params.min_prominence_frac:
            levels.append(Level(j + w, j, float(lows[j]), "support"))
    return levels


def _merge(levels: Sequence[Level], tolerance: float) -> list[tuple[float, str, int]]:
    """Collapse levels of the same kind that sit within ``tolerance`` of each other."""
    if not levels:
        return []
    ordered = sorted(levels, key=lambda lv: (lv.kind, lv.price))
    merged: list[tuple[float, str, int]] = []
    cur_price, cur_kind, touches = ordered[0].price, ordered[0].kind, 1

    for lv in ordered[1:]:
        if lv.kind == cur_kind and abs(lv.price - cur_price) <= tolerance:
            cur_price = (cur_price * touches + lv.price) / (touches + 1)
            touches += 1
        else:
            merged.append((cur_price, cur_kind, touches))
            cur_price, cur_kind, touches = lv.price, lv.kind, 1
    merged.append((cur_price, cur_kind, touches))
    return merged


def merged_levels(frame: "StrategyFrame", i: Optional[int] = None) -> list[tuple[float, str, int]]:
    """Support and resistance as the engine sees them at bar ``i``.

    Returns ``(price, kind, touches)``. Exposed so the charts can label exactly the
    levels the entry rule is comparing against — a chart drawing different levels
    from the ones the signal uses is worse than a chart with no levels on it.
    """
    if i is None:
        i = len(frame) - 1
    if i < 0:
        return []
    atr = float(frame.df.iloc[i]["ATR"])
    tolerance = max(1e-9, frame.params.level_merge_atr_mult * atr)
    return _merge(frame.levels_asof(i), tolerance)


def prepare(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> StrategyFrame:
    """Compute everything the strategy needs, once, for a whole price history."""
    annotated = add_indicators(df, params)
    return StrategyFrame(annotated, detect_levels(annotated, params), params)


def drop_forming_bar(df: pd.DataFrame, asset_class: AssetClass, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Remove a still-forming final bar.

    The exit rule is defined on the *close*. Evaluating it against a bar that is
    still printing makes the signal repaint — it fires mid-session and unfires
    before the bell. Crypto has no session boundary, so the last daily bar is
    always incomplete until the date rolls over.
    """
    if df.empty:
        return df
    now = pd.Timestamp.now("UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    last = pd.Timestamp(df.index[-1])
    if getattr(last, "tzinfo", None) is not None:
        last = last.tz_localize(None)
    if last.normalize() >= now.normalize():
        return df.iloc[:-1]
    return df


def evaluate(
    frame: StrategyFrame,
    i: int,
    position: Position = Position(),
    regime_ok: bool = True,
) -> Decision:
    """Decide what to do at bar ``i`` using only bars ``0..i``."""
    params = frame.params
    row = frame.df.iloc[i]
    price = float(row["Close"])
    sma = float(row[f"SMA_{params.sma_exit}"])
    atr = float(row["ATR"])
    logs: list[str] = []

    if np.isnan(sma):
        return Decision(Action.HOLD, price, sma, atr, regime_ok=regime_ok,
                        logs=("Insufficient history for the exit moving average.",))

    tolerance = max(1e-9, params.level_merge_atr_mult * atr)
    merged = _merge(frame.levels_asof(i), tolerance)
    resistances = sorted([p for p, kind, _ in merged if kind == "resistance"])
    broken = next((p for p in reversed(resistances) if p < price), None)
    overhead = next((p for p in resistances if p > price), None)

    # Rule 1 - trailing exit. Returns unconditionally: a name trading below its
    # exit average is never a fresh entry, long-term flag or not. The original
    # code fell through to the breakout block when long_term was set, which let a
    # holding in a confirmed downtrend still emit BUY.
    if price < sma:
        if position.long_term:
            logs.append(
                f"Price ${price:,.2f} is below the {params.sma_exit} SMA ${sma:,.2f}; "
                "holding under the long-term flag. No new entries."
            )
            return Decision(Action.HOLD, price, sma, atr, broken, overhead, regime_ok, tuple(logs))

        logs.append(f"EXIT: close ${price:,.2f} below the {params.sma_exit} SMA ${sma:,.2f}.")
        if position.is_open:
            logs.append("Action: liquidate the position.")
            return Decision(Action.SELL, price, sma, atr, broken, overhead, regime_ok, tuple(logs))
        logs.append("Flat already; nothing to do.")
        return Decision(Action.HOLD, price, sma, atr, broken, overhead, regime_ok, tuple(logs))

    # Rule 2 - breakout entry.
    if overhead is not None:
        logs.append(f"Overhead resistance at ${overhead:,.2f}.")

    if broken is None:
        logs.append("No confirmed resistance below price yet.")
        return Decision(Action.HOLD, price, sma, atr, broken, overhead, regime_ok, tuple(logs))

    extension = (price - broken) / broken
    if not (0 < extension < params.breakout_band):
        logs.append(
            f"Holding. Price sits {extension * 100:.2f}% above the broken level "
            f"${broken:,.2f}, outside the {params.breakout_band * 100:.0f}% entry band."
        )
        return Decision(Action.HOLD, price, sma, atr, broken, overhead, regime_ok, tuple(logs))

    logs.append(f"Fresh breakout: {extension * 100:.2f}% above ${broken:,.2f}.")

    if position.is_open:
        logs.append("Already positioned; no add.")
        return Decision(Action.HOLD, price, sma, atr, broken, overhead, regime_ok, tuple(logs))

    if not regime_ok:
        logs.append("Entry vetoed: market regime is risk-off.")
        return Decision(Action.HOLD, price, sma, atr, broken, overhead, regime_ok, tuple(logs))

    logs.append("BUY signal.")
    return Decision(Action.BUY, price, sma, atr, broken, overhead, regime_ok, tuple(logs))


def evaluate_latest(
    df: pd.DataFrame,
    position: Position = Position(),
    params: StrategyParams = StrategyParams(),
    regime_ok: bool = True,
) -> tuple[StrategyFrame, Decision]:
    """Convenience wrapper for the live path: decide on the most recent bar."""
    frame = prepare(df, params)
    if len(frame) == 0:
        raise ValueError("empty price history")
    return frame, evaluate(frame, len(frame) - 1, position, regime_ok)
