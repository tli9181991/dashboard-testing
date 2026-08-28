"""Market regime gate.

The breakout strategy is long-only and structurally short volatility: it bleeds in
chop and takes its worst losses in sustained downtrends. Gating *new entries* on
the broad market holding above its own long moving average removes most of that
exposure at the cost of missing the first leg off a bottom.

Exits are never gated. A risk-off reading must not trap an open position.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeParams:
    sma: int = 200
    #: Benchmark whose trend defines risk-on. ^GSPC keeps it consistent with the screener.
    benchmark: str = "^GSPC"


def regime_series(benchmark_close: pd.Series, params: RegimeParams = RegimeParams()) -> pd.Series:
    """True on days the benchmark closed above its moving average.

    Rolling means are backward-looking, so the value at any date uses only that
    date and earlier. Warmup bars are False: unknown regime means no new risk.
    """
    close = pd.Series(benchmark_close).astype(float).dropna()
    sma = close.rolling(params.sma).mean()
    return (close > sma).fillna(False)


def align_regime(regime: pd.Series, index: pd.Index) -> pd.Series:
    """Project a regime series onto another instrument's calendar.

    Crypto prints on weekends and holidays when the equity benchmark does not, so
    the last known reading is carried forward. Forward-fill only looks backward,
    which keeps the alignment causal. Dates before the first reading are False.
    """
    if regime.empty:
        return pd.Series(False, index=index, dtype=bool)
    combined = regime.reindex(regime.index.union(index)).ffill()
    return combined.reindex(index).fillna(False).astype(bool)


def build_gate(benchmark_close: pd.Series, index: pd.Index,
               params: RegimeParams = RegimeParams()) -> pd.Series:
    """Convenience: regime series aligned to a symbol's index in one call."""
    return align_regime(regime_series(benchmark_close, params), index)
