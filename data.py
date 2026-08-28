"""Price history loading for research and backtesting.

Order of preference: an explicit CSV cache on disk, then yfinance if it is
installed and reachable. ``synthetic_ohlcv`` generates deterministic data so the
test suite and a first backtest run offline.

yfinance is fine for research and deliberately *not* wired into anything that
places orders: it is an unofficial endpoint, it returns empty frames when
throttled, and it restates history after splits and dividends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

CACHE_DIR = Path("./.screen_cache")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(symbol: str, period: str) -> Path:
    return CACHE_DIR / f"bt_{symbol.replace('-', '_').replace('^', '')}_{period}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]
    keep = [c for c in OHLCV if c in df.columns]
    out = df[keep].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def load_history(symbol: str, period: str = "5y", use_cache: bool = True,
                 cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Daily OHLCV for one symbol. Returns an empty frame if unavailable."""
    directory = cache_dir or CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _cache_path(symbol, period).name

    if use_cache and path.exists():
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            if not cached.empty:
                return _normalize(cached)
        except Exception:
            pass

    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(columns=OHLCV)

    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          progress=False, auto_adjust=True)
    except Exception:
        return pd.DataFrame(columns=OHLCV)

    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV)

    frame = _normalize(raw)
    if not frame.empty:
        frame.to_csv(path)
    return frame


def load_universe(symbols: Iterable[str], period: str = "5y",
                  use_cache: bool = True) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        frame = load_history(sym, period=period, use_cache=use_cache)
        if not frame.empty:
            out[sym] = frame
    return out


def synthetic_ohlcv(
    n: int = 900,
    seed: int = 7,
    start: str = "2021-01-04",
    initial_price: float = 100.0,
    annual_drift: float = 0.10,
    annual_vol: float = 0.32,
    regime_flip: Optional[int] = None,
) -> pd.DataFrame:
    """Deterministic geometric-random-walk OHLCV with business-day stamps.

    Used by the tests and by ``backtest.py --demo`` so the machinery can be
    exercised without a network call. It is a sanity harness, not a market: a
    result produced on this data says nothing about the strategy's edge.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = np.full(n, annual_drift)
    if regime_flip is not None and 0 < regime_flip < n:
        mu[regime_flip:] = -abs(annual_drift) * 1.5

    shocks = rng.normal(0.0, annual_vol * np.sqrt(dt), size=n)
    log_returns = (mu - 0.5 * annual_vol**2) * dt + shocks
    close = initial_price * np.exp(np.cumsum(log_returns))

    intrabar = np.abs(rng.normal(0.0, annual_vol * np.sqrt(dt) * 0.8, size=n))
    open_ = np.empty(n)
    open_[0] = initial_price
    open_[1:] = close[:-1] * (1 + rng.normal(0.0, 0.002, size=n - 1))
    high = np.maximum(open_, close) * (1 + intrabar)
    low = np.minimum(open_, close) * (1 - intrabar)
    volume = rng.integers(500_000, 5_000_000, size=n).astype(float)

    index = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def synthetic_benchmark(index: pd.Index, seed: int = 11, annual_vol: float = 0.16,
                        annual_drift: float = 0.08, regime_flip: Optional[int] = None) -> pd.Series:
    """A benchmark close series on a given calendar, for regime testing."""
    frame = synthetic_ohlcv(
        n=len(index), seed=seed, annual_vol=annual_vol,
        annual_drift=annual_drift, regime_flip=regime_flip,
    )
    return pd.Series(frame["Close"].to_numpy(), index=index, name="benchmark")
