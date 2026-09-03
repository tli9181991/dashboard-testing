"""Coverage for the Swing Universe Funnel.

The load-bearing test here is the swing-point lag: `swing_points` shifts both
series by one bar because a fractal at bar i is only knowable at bar i+1. Remove
that shift and every downstream ICT signal is reading the future.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import swing_screener as ss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _uptrend_with_pullback(n=320, base=100.0, daily=0.0015, noise=0.0016,
                           pull_len=3, seed=5):
    """A clean uptrend, a quiet contracting pullback, then an up-close trigger.

    Built to satisfy §04A rather than sampled at random: the momentum setup needs
    a trend template, a shallow pullback onto a short EMA, drying volume and
    contracting ranges all at once, which random walks essentially never produce.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=n)
    close = base * np.exp(np.cumsum(rng.normal(daily, noise, n)))

    peak = close[-(pull_len + 2)]
    for k in range(pull_len):
        close[-(pull_len + 1) + k] = peak * (1 - 0.0035 * (k + 1))
    close[-1] = close[-2] * 1.010

    openp = np.concatenate([[close[0]], close[:-1]])
    spread = close * 0.004
    high = np.maximum(openp, close) + spread
    low = np.minimum(openp, close) - spread
    for k in range(pull_len):
        j = -(pull_len + 1) + k
        mid, half = (high[j] + low[j]) / 2, (high[j] - low[j]) / 2 * (0.85 ** (k + 1))
        high[j], low[j] = mid + half, mid - half

    vol = np.full(n, 3_000_000.0)
    vol[-(pull_len + 1):-1] = 900_000.0
    return pd.DataFrame({"Open": openp, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def _flat(n=300, price=50.0, volume=5_000_000.0):
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=n)
    return pd.DataFrame({"Open": price, "High": price * 1.01, "Low": price * 0.99,
                         "Close": price, "Volume": volume}, index=idx)


@pytest.fixture(scope="module")
def cfg():
    return dict(ss.CFG)


@pytest.fixture(scope="module")
def demo():
    bars = ss.load_demo(n_names=60, seed=11)
    spy = bars.pop("SPY")
    return bars, spy


# ---------------------------------------------------------------------------
# Causality — the reason any of the rest can be trusted
# ---------------------------------------------------------------------------

def test_swing_points_are_lagged_by_one_bar():
    """A fractal high at bar i must not be flagged until bar i+1."""
    highs = [10, 11, 12, 20, 12, 11, 10, 9, 8, 9, 10]
    lows = [h - 2 for h in highs]
    idx = pd.bdate_range("2026-01-01", periods=len(highs))
    df = pd.DataFrame({"Open": highs, "High": highs, "Low": lows,
                       "Close": highs, "Volume": 1e6}, index=idx)

    sh, _ = ss.swing_points(df)
    peak = 3                                  # the value-20 bar
    assert not bool(sh.iloc[peak]), "swing flagged on the bar itself — reads the future"
    assert bool(sh.iloc[peak + 1]), "swing should become known one bar later"


def test_swing_points_do_not_change_when_later_bars_arrive():
    df = _uptrend_with_pullback()
    full, _ = ss.swing_points(df)
    for cutoff in (150, 220, 300):
        trunc, _ = ss.swing_points(df.iloc[:cutoff])
        pd.testing.assert_series_equal(
            full.iloc[:cutoff - 1], trunc.iloc[:cutoff - 1], check_names=False
        )


def test_prior_period_levels_use_only_completed_periods():
    """The week in progress must not leak into its own levels."""
    df = _uptrend_with_pullback()
    levels = ss.prior_period_levels(df)
    # Every level is drawn from a prior period, so it cannot equal a high set today.
    assert levels["wk_hi"].iloc[-1] <= float(df["High"].max())
    trunc = ss.prior_period_levels(df.iloc[:-5])
    common = levels.index.intersection(trunc.index)[:-5]
    pd.testing.assert_frame_equal(levels.loc[common], trunc.loc[common])


# ---------------------------------------------------------------------------
# §01 liquidity gate
# ---------------------------------------------------------------------------

def test_gate_rejects_short_history(cfg):
    assert ss.liquidity_gate(_flat(n=50), cfg)[0] is False


def test_gate_rejects_penny_prices(cfg):
    ok, why = ss.liquidity_gate(_flat(price=3.0), cfg)
    assert ok is False and "price" in why


def test_gate_rejects_thin_dollar_volume(cfg):
    ok, why = ss.liquidity_gate(_flat(price=50.0, volume=1_000.0), cfg)
    assert ok is False and "dollar vol" in why


def test_gate_passes_a_liquid_name(cfg):
    assert ss.liquidity_gate(_flat(price=50.0, volume=5_000_000.0), cfg)[0] is True


@pytest.mark.parametrize("meta,expected", [
    ({"market_cap": 100e6}, "market cap"),
    ({"float_shares": 1e6}, "float"),
    ({"spread_bps": 50}, "spread"),
    ({"merger_target": True}, "merger_target"),
    ({"leveraged_etf": True}, "leveraged_etf"),
])
def test_gate_honours_vendor_metadata(cfg, meta, expected):
    ok, why = ss.liquidity_gate(_flat(price=50.0, volume=5_000_000.0), cfg, meta)
    assert ok is False and why == expected


def test_position_is_capped_by_average_volume(cfg):
    df = _flat(volume=1_000_000.0)
    assert ss.max_shares_by_liquidity(df, cfg) == int(1_000_000 * cfg["max_pct_of_adv"])


# ---------------------------------------------------------------------------
# §03 regime
# ---------------------------------------------------------------------------

def _spy(trend):
    n = 300
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=n)
    close = np.linspace(100, 100 * trend, n)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": 1e8}, index=idx)


def test_downtrend_is_risk_off(cfg):
    assert ss.regime_state(_spy(0.6), breadth=0.6, vix=15, cfg=cfg) == "risk_off"


def test_thin_breadth_is_risk_off(cfg):
    assert ss.regime_state(_spy(1.5), breadth=0.20, vix=15, cfg=cfg) == "risk_off"


def test_high_vix_is_risk_off(cfg):
    assert ss.regime_state(_spy(1.5), breadth=0.8, vix=35, cfg=cfg) == "risk_off"


def test_healthy_tape_is_risk_on(cfg):
    assert ss.regime_state(_spy(1.5), breadth=0.8, vix=15, cfg=cfg) == "risk_on"


def test_risk_off_allows_no_positions_and_zero_size(cfg):
    assert ss.positions_allowed("risk_off", cfg) == 0
    assert ss.size_multiplier("risk_off") == 0.0
    assert ss.size_multiplier("mixed") == 0.5
    assert ss.positions_allowed("risk_on", cfg) == cfg["max_positions"]


def test_sector_ranks_order_strongest_first(cfg):
    def series(mult):
        n = 100
        idx = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=n)
        close = np.linspace(100, 100 * mult, n)
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": 1e7}, index=idx)

    ranks = ss.sector_ranks({"XLK": series(1.5), "XLU": series(1.0), "XLE": series(1.2)}, cfg)
    assert ranks["XLK"] == 1 and ranks["XLE"] == 2 and ranks["XLU"] == 3


# ---------------------------------------------------------------------------
# §04 setups
# ---------------------------------------------------------------------------

def test_momentum_setup_fires_on_a_textbook_pullback(cfg):
    setup = ss.momentum_setup(_uptrend_with_pullback(), cfg)
    assert setup is not None
    assert setup["variant"] == "A/momentum"
    assert setup["stop"] < setup["entry"] < setup["tp1"] <= setup["tp2"]
    assert "pullback" in setup["note"]


def test_momentum_setup_needs_an_up_close(cfg):
    df = _uptrend_with_pullback()
    df.iloc[-1, df.columns.get_loc("Close")] = float(df["Close"].iloc[-2]) * 0.99
    assert ss.momentum_setup(df, cfg) is None


def test_momentum_setup_rejects_a_pullback_on_heavy_volume(cfg):
    """Distribution, not a pause: same price action, volume never dries up."""
    df = _uptrend_with_pullback()
    df["Volume"] = 3_000_000.0
    assert ss.momentum_setup(df, cfg) is None


def test_momentum_setup_requires_the_trend_template(cfg):
    assert ss.momentum_setup(_flat(), cfg) is None


def test_trend_template_rejects_a_flat_series():
    assert ss.trend_template(_flat()) is False


def test_trend_template_accepts_a_clean_uptrend():
    assert ss.trend_template(_uptrend_with_pullback()) is True


def test_bullish_fvg_detects_a_three_bar_imbalance():
    idx = pd.bdate_range("2026-01-01", periods=3)
    df = pd.DataFrame({"Open": [10, 11, 13], "High": [10.5, 12, 14],
                       "Low": [9.5, 10.5, 11.0], "Close": [10, 12, 13.5],
                       "Volume": 1e6}, index=idx)
    gap = ss.bullish_fvg(df, 2)
    assert gap == (10.5, 11.0)          # High[0] .. Low[2]


def test_bullish_fvg_absent_when_bars_overlap():
    idx = pd.bdate_range("2026-01-01", periods=3)
    df = pd.DataFrame({"Open": [10, 10, 10], "High": [12, 12, 12],
                       "Low": [9, 9, 9], "Close": [10, 10, 10],
                       "Volume": 1e6}, index=idx)
    assert ss.bullish_fvg(df, 2) is None


def test_ict_setup_fires_somewhere_in_the_demo_universe(cfg, demo):
    bars, _ = demo
    setups = [ss.ict_setup(d, cfg) for d in bars.values()]
    fired = [s for s in setups if s]
    assert fired, "fixture produced no ICT setups; the test proves nothing"
    for s in fired:
        assert s["variant"] == "B/ict"
        assert s["stop"] < s["entry"] <= s["tp1"] <= s["tp2"]


# ---------------------------------------------------------------------------
# §05 events
# ---------------------------------------------------------------------------

def test_earnings_inside_the_blackout_vetoes(cfg):
    asof = date(2026, 6, 30)
    veto = ss.earnings_veto("AAA", asof, {"AAA": (asof + timedelta(days=3), True)}, cfg)
    assert "earnings" in veto


def test_earnings_far_away_does_not_veto(cfg):
    asof = date(2026, 6, 30)
    assert ss.earnings_veto("AAA", asof, {"AAA": (asof + timedelta(days=60), True)}, cfg) == ""


def test_an_estimated_date_is_buffered_wider_than_a_confirmed_one(cfg):
    asof = date(2026, 6, 30)
    edge = asof + timedelta(days=9)
    assert ss.earnings_veto("AAA", asof, {"AAA": (edge, True)}, cfg) == ""
    assert ss.earnings_veto("AAA", asof, {"AAA": (edge, False)}, cfg) != ""


def test_unknown_ticker_is_not_vetoed(cfg):
    assert ss.earnings_veto("ZZZ", date(2026, 6, 30), {}, cfg) == ""


# ---------------------------------------------------------------------------
# §06 sizing
# ---------------------------------------------------------------------------

def test_size_follows_the_stop(cfg):
    df = _flat(volume=100_000_000.0)
    wide = ss.position_size(100.0, 95.0, df, "risk_on", cfg)
    tight = ss.position_size(100.0, 99.0, df, "risk_on", cfg)
    assert tight > wide
    assert wide == int(cfg["account_equity"] * cfg["risk_per_trade"] / 5.0)


def test_mixed_regime_halves_the_size(cfg):
    df = _flat(volume=100_000_000.0)
    full = ss.position_size(100.0, 95.0, df, "risk_on", cfg)
    assert ss.position_size(100.0, 95.0, df, "mixed", cfg) == full // 2


def test_risk_off_sizes_to_zero(cfg):
    assert ss.position_size(100.0, 95.0, _flat(), "risk_off", cfg) == 0


def test_an_inverted_stop_sizes_to_zero(cfg):
    assert ss.position_size(100.0, 105.0, _flat(), "risk_on", cfg) == 0


def test_size_is_capped_by_liquidity(cfg):
    thin = _flat(volume=1_000.0)
    assert ss.position_size(100.0, 99.9, thin, "risk_on", cfg) <= \
        ss.max_shares_by_liquidity(thin, cfg)


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def test_scan_reports_the_full_funnel(cfg, demo):
    bars, spy = demo
    out, ctx = ss.run_scan(bars, spy, cfg)
    for key in ("asof", "universe", "gated", "floor_ok", "tradable",
                "breadth", "regime", "slots", "stage"):
        assert key in ctx, key
    assert ctx["universe"] >= ctx["gated"] >= ctx["floor_ok"] >= ctx["tradable"]
    assert isinstance(out, pd.DataFrame)


def test_an_empty_gate_still_reports_a_regime(cfg, demo):
    """Regression: callers read ctx['regime'] unconditionally."""
    _, spy = demo
    out, ctx = ss.run_scan({"AAA": _flat(n=10)}, spy, cfg)
    assert out.empty
    assert ctx["regime"] == "unknown"
    assert ctx["slots"] == 0
    assert ctx["stage"] == "gate"


def test_risk_off_returns_no_candidates(cfg, demo):
    bars, _ = demo
    out, ctx = ss.run_scan(bars, _spy(0.6), cfg)
    assert out.empty
    assert ctx["regime"] == "risk_off"
    assert ctx["slots"] == 0


def test_candidates_carry_a_complete_trade_plan(cfg):
    bars = ss.load_demo(n_names=200, seed=11)
    spy = bars.pop("SPY")
    out, ctx = ss.run_scan(bars, spy, cfg)
    if out.empty:
        pytest.skip("no setups in this synthetic universe")
    assert (out["stop"] < out["entry"]).all()
    assert (out["entry"] <= out["tp1"]).all()
    assert (out["tp1"] <= out["tp2"]).all()
    assert (out["shares"] >= 1).all()
    assert (out["risk_$"] > 0).all()
    assert out["score"].is_monotonic_decreasing


def test_demo_volume_tracks_price_activity():
    """Regression: i.i.d. volume made the §04A dry-up test unsatisfiable."""
    bars = ss.load_demo(n_names=20, seed=3)
    bars.pop("SPY")
    correlations = []
    for d in bars.values():
        move = d["Close"].pct_change().abs()
        correlations.append(move.corr(d["Volume"]))
    assert np.nanmean(correlations) > 0.2, "volume is independent of price again"
