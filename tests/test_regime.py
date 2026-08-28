import numpy as np
import pandas as pd

from regime import RegimeParams, align_regime, build_gate, regime_series


def test_regime_is_true_above_the_average_and_false_below():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rising = pd.Series(np.linspace(100, 200, 400), index=idx)
    gate = regime_series(rising, RegimeParams(sma=200))
    assert gate.iloc[-1]

    falling = pd.Series(np.linspace(200, 100, 400), index=idx)
    assert not regime_series(falling, RegimeParams(sma=200)).iloc[-1]


def test_warmup_bars_are_risk_off():
    idx = pd.bdate_range("2020-01-01", periods=400)
    series = pd.Series(np.linspace(100, 200, 400), index=idx)
    gate = regime_series(series, RegimeParams(sma=200))
    assert not gate.iloc[:199].any(), "regime must default to risk-off before it is defined"


def test_regime_uses_no_future_data():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    series = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)), index=idx)
    full = regime_series(series, RegimeParams(sma=200))
    trunc = regime_series(series.iloc[:300], RegimeParams(sma=200))
    pd.testing.assert_series_equal(full.iloc[:300], trunc)


def test_alignment_forward_fills_onto_a_247_calendar():
    """Crypto trades weekends; the benchmark does not. Carry the last known reading."""
    bench_idx = pd.bdate_range("2023-01-02", periods=10)
    crypto_idx = pd.date_range("2023-01-02", periods=14, freq="D")
    gate = pd.Series([True] * 10, index=bench_idx)

    aligned = align_regime(gate, crypto_idx)
    assert len(aligned) == 14
    assert aligned.loc["2023-01-07"]  # a Saturday
    assert aligned.dtype == bool


def test_alignment_is_risk_off_before_the_first_reading():
    bench_idx = pd.bdate_range("2023-02-01", periods=5)
    target_idx = pd.date_range("2023-01-01", periods=40, freq="D")
    aligned = align_regime(pd.Series([True] * 5, index=bench_idx), target_idx)
    assert not aligned.iloc[0]


def test_build_gate_end_to_end(prices, benchmark):
    gate = build_gate(benchmark, prices.index)
    assert len(gate) == len(prices)
    assert gate.dtype == bool
    assert 0 < gate.sum() < len(gate), "fixture should contain both regimes"
