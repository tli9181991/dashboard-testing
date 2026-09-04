"""Agent tools: the point is that the model can compute, and can be refused."""

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import agent_tools as at
import data as data_mod


@pytest.fixture(scope="module")
def prices():
    return data_mod.synthetic_ohlcv(n=700, seed=42)


@pytest.fixture
def loaded(prices, monkeypatch):
    bench = data_mod.synthetic_ohlcv(n=700, seed=5, annual_vol=0.16)

    def fake(symbol, period="3y", **kwargs):
        return bench if symbol == "^GSPC" else prices

    monkeypatch.setattr(at.data_mod, "load_history", fake)
    return prices


@pytest.fixture
def no_prices(monkeypatch):
    monkeypatch.setattr(at.data_mod, "load_history",
                        lambda *a, **k: pd.DataFrame())


def _last_price(frame):
    import strategy as strat
    return float(strat.prepare(frame).df["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# 1. Earnings
# ---------------------------------------------------------------------------

class _Handle:
    def __init__(self, calendar=None, table=None):
        self.calendar = calendar
        self.earnings_dates = table


def _patch_yf(monkeypatch, handle):
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda symbol: handle)


def test_a_report_inside_the_window_is_a_blackout(monkeypatch):
    asof = date(2026, 6, 1)
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2026, 6, 4)]}))
    info = at.earnings_calendar("AAA", asof=asof)
    assert info["in_blackout"] is True
    assert info["days_away"] == 3
    assert "BLACKOUT" in at.render_earnings(info)


def test_a_distant_report_is_clear(monkeypatch):
    asof = date(2026, 6, 1)
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2026, 8, 1)]}))
    info = at.earnings_calendar("AAA", asof=asof)
    assert info["in_blackout"] is False
    assert "clear" in at.render_earnings(info)


def test_an_estimated_date_widens_the_window(monkeypatch):
    """Vendors are routinely a week out, so a range buys more caution, not less."""
    asof = date(2026, 6, 1)
    edge = date(2026, 6, 10)

    _patch_yf(monkeypatch, _Handle({"Earnings Date": [edge]}))
    confirmed = at.earnings_calendar("AAA", asof=asof)

    _patch_yf(monkeypatch, _Handle({"Earnings Date": [edge, date(2026, 6, 12)]}))
    estimated = at.earnings_calendar("AAA", asof=asof)

    assert confirmed["confirmed"] is True
    assert estimated["confirmed"] is False
    assert estimated["blackout_before_days"] > confirmed["blackout_before_days"]


def test_a_recent_report_is_also_a_blackout(monkeypatch):
    asof = date(2026, 6, 10)
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2026, 6, 9),
                                                      date(2026, 9, 9)]}))
    info = at.earnings_calendar("AAA", asof=asof)
    assert info["days_since_last"] == 1
    assert info["in_blackout"] is True


def test_the_earnings_dates_table_is_used_as_a_fallback(monkeypatch):
    table = pd.DataFrame({"EPS Estimate": [1.0]},
                         index=pd.DatetimeIndex([datetime(2026, 6, 3)]))
    _patch_yf(monkeypatch, _Handle(calendar=None, table=table))
    info = at.earnings_calendar("AAA", asof=date(2026, 6, 1))
    assert info["next_date"] == "2026-06-03"


def test_no_published_date_is_reported_not_guessed(monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": []}))
    info = at.earnings_calendar("AAA", asof=date(2026, 6, 1))
    assert info["next_date"] is None
    assert info["in_blackout"] is False
    assert "No future earnings" in at.render_earnings(info)


def test_a_lookup_failure_is_reported_not_raised(monkeypatch):
    import yfinance as yf

    def boom(symbol):
        raise RuntimeError("network down")

    monkeypatch.setattr(yf, "Ticker", boom)
    info = at.earnings_calendar("AAA")
    assert info["error"]
    assert "network down" in at.render_earnings(info)


def test_earnings_map_is_shaped_for_the_screener_veto(monkeypatch):
    """swing_screener.earnings_veto expects {ticker: (date, confirmed)}."""
    import swing_screener as swing

    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2026, 6, 4)]}))
    mapping = at.earnings_map(["aaa"], asof=date(2026, 6, 1))
    assert mapping["AAA"][0] == date(2026, 6, 4)
    assert mapping["AAA"][1] is True
    assert swing.earnings_veto("AAA", date(2026, 6, 1), mapping, dict(swing.CFG))


# ---------------------------------------------------------------------------
# 2. Trade plan validation — refusal
# ---------------------------------------------------------------------------

def test_a_sound_plan_passes_every_check(loaded, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px * 1.01, stop=px * 0.985,
                                    target=px * 1.08)
    assert result["verdict"] == "PASS", result["failed"]
    assert result["shares"] >= 1
    assert result["dollar_risk"] > 0


def test_a_stop_above_the_entry_is_refused(loaded, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px, stop=px * 1.05)
    assert result["verdict"] == "FAIL"
    assert "stop below entry" in result["failed"]


def test_a_stop_beyond_the_atr_budget_is_refused(loaded, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px, stop=px * 0.5)
    assert result["verdict"] == "FAIL"
    assert "stop within ATR budget" in result["failed"]


def test_a_thin_reward_to_risk_is_refused(loaded, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px, stop=px * 0.98,
                                    target=px * 1.001)
    assert result["verdict"] == "FAIL"
    assert "reward:risk" in result["failed"]


def test_an_earnings_blackout_refuses_the_plan(loaded, monkeypatch):
    """The check that most often invalidates a swing trade."""
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date.today() + timedelta(days=2)]}))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px * 1.01, stop=px * 0.985,
                                    target=px * 1.08)
    assert result["verdict"] == "FAIL"
    assert "earnings blackout" in result["failed"]


def test_an_unknown_earnings_date_does_not_block_the_trade(loaded, monkeypatch):
    """Missing vendor data is not evidence of a blackout."""
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda s: (_ for _ in ()).throw(RuntimeError("no net")))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px * 1.01, stop=px * 0.985,
                                    target=px * 1.08)
    assert "earnings blackout" not in result["failed"]
    detail = [c for c in result["checks"] if c["name"] == "earnings blackout"][0]
    assert "unknown" in detail["detail"]


def test_the_position_cap_is_enforced(loaded, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    px = _last_price(loaded)
    result = at.validate_trade_plan("TEST", entry=px, stop=px * 0.99,
                                    target=px * 1.05, equity=50_000,
                                    max_position_pct=0.10)
    assert result["notional"] <= 50_000 * 0.10 + 1e-6


def test_the_rendered_refusal_names_the_broken_rule(loaded, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    px = _last_price(loaded)
    text = at.render_validation(at.validate_trade_plan("TEST", entry=px, stop=px * 1.05))
    assert "FAIL" in text
    assert "stop below entry" in text


def test_validation_without_price_history_fails_loudly(no_prices, monkeypatch):
    _patch_yf(monkeypatch, _Handle({"Earnings Date": [date(2030, 1, 1)]}))
    result = at.validate_trade_plan("ZZZ", entry=100, stop=95, target=115)
    assert result["verdict"] == "FAIL"
    assert "price history" in result["failed"]


# ---------------------------------------------------------------------------
# 3. Signal, levels, size
# ---------------------------------------------------------------------------

def test_the_signal_comes_from_the_engine_not_a_guess(loaded):
    info = at.check_signal_now("TEST")
    assert info["action"] in {"BUY", "SELL", "HOLD"}
    assert info["logs"], "the engine's own reasoning must come back with it"
    assert "last CLOSED bar" in at.render_signal(info)


def test_the_signal_matches_calling_the_engine_directly(loaded):
    import strategy as strat
    closed = strat.drop_forming_bar(loaded, strat.AssetClass.EQUITY)
    _, decision = strat.evaluate_latest(closed if not closed.empty else loaded,
                                        strat.Position(), strat.StrategyParams(),
                                        at.check_signal_now("TEST")["regime_ok"])
    assert at.check_signal_now("TEST")["action"] == decision.action.value


def test_a_risk_off_regime_is_reported(loaded, monkeypatch):
    falling = data_mod.synthetic_ohlcv(n=700, seed=3, annual_drift=-0.5)
    monkeypatch.setattr(at.data_mod, "load_history",
                        lambda s, period="3y", **k: falling if s == "^GSPC" else loaded)
    info = at.check_signal_now("TEST")
    assert info["regime_ok"] is False
    assert "risk-off" in info["regime_note"]


def test_levels_are_split_around_the_current_price(loaded):
    info = at.support_resistance("TEST")
    for row in info["resistance"]:
        assert row["price"] > info["price"]
        assert row["distance_pct"] > 0
    for row in info["support"]:
        assert row["price"] < info["price"]
        assert row["distance_pct"] < 0


def test_levels_match_the_engines_own_detection(loaded):
    import strategy as strat
    frame = strat.prepare(strat.drop_forming_bar(loaded, strat.AssetClass.EQUITY))
    engine = {round(p, 6) for p, _, _ in strat.merged_levels(frame)}
    info = at.support_resistance("TEST")
    reported = {round(r["price"], 6) for r in info["resistance"] + info["support"]}
    assert reported <= engine


def test_size_scales_inversely_with_the_vol_target(loaded):
    small = at.size_position("TEST", equity=100_000, target_vol=0.05,
                             max_position_pct=1.0)
    large = at.size_position("TEST", equity=100_000, target_vol=0.30,
                             max_position_pct=1.0)
    assert large["quantity"] > small["quantity"]
    assert large["quantity"] == pytest.approx(small["quantity"] * 6, rel=0.05)


def test_size_respects_the_cap_and_says_when_it_binds(loaded):
    """Regression: flooring to whole shares left the notional just under the cap,
    so comparing the two made `capped` read False exactly when it was true."""
    info = at.size_position("TEST", equity=100_000, target_vol=0.90,
                            max_position_pct=0.10)
    assert info["notional"] <= 100_000 * 0.10 + 1e-6
    assert info["uncapped_notional"] > 100_000 * 0.10
    assert info["capped"] is True
    assert "cap binding" in at.render_size(info)


def test_an_unbound_size_is_not_reported_as_capped(loaded):
    info = at.size_position("TEST", equity=100_000, target_vol=0.05,
                            max_position_pct=1.0)
    assert info["capped"] is False
    assert "cap binding" not in at.render_size(info)


@pytest.mark.parametrize("call", ["check_signal_now", "support_resistance", "size_position"])
def test_every_tool_reports_missing_data_rather_than_raising(no_prices, call):
    info = getattr(at, call)("ZZZ")
    assert info["error"]


# ---------------------------------------------------------------------------
# 4. Random-entry test
# ---------------------------------------------------------------------------

def test_the_random_entry_test_places_the_rule_in_a_distribution(loaded):
    info = at.random_entry_test("TEST", n_paths=120)
    assert info["error"] == ""
    if info["n_trades"]:
        assert 0.0 <= info["percentile"] <= 100.0
        assert info["actual_mean_trade"] is not None


def test_it_says_so_when_a_weak_rule_fails_to_beat_random(loaded):
    """The finding a language model will never volunteer on its own."""
    info = at.random_entry_test("TEST", n_paths=120)
    if info["n_trades"] and info["percentile"] < 60:
        assert "NOT beating random entry" in at.render_random_entry(info)


def test_it_always_flags_the_single_symbol_caveat(loaded):
    info = at.random_entry_test("TEST", n_paths=120)
    if info["n_trades"]:
        assert "characterisation, not an edge estimate" in at.render_random_entry(info)


def test_no_trades_is_reported_rather_than_scored(monkeypatch):
    flat = data_mod.synthetic_ohlcv(n=400, seed=1, annual_vol=0.0001, annual_drift=0.0)
    monkeypatch.setattr(at.data_mod, "load_history", lambda *a, **k: flat)
    info = at.random_entry_test("FLAT", n_paths=50)
    assert info["n_trades"] == 0
    assert "nothing to compare" in at.render_random_entry(info)


def test_it_reports_missing_data_rather_than_raising(no_prices):
    info = at.random_entry_test("ZZZ", n_paths=50)
    assert info["error"]


# ---------------------------------------------------------------------------
# The screener's §05 veto, now that it has a data source
# ---------------------------------------------------------------------------

def test_the_screener_veto_removes_names_inside_the_window():
    """Regression: run_scan defaulted `earnings` to {} and nothing ever supplied
    it, so the blackout was a no-op and the screener would happily surface a
    setup reporting in two days."""
    from datetime import timedelta
    import swing_screener as swing

    found = None
    for seed in (11, 3, 7):
        for keep in (0.6, 0.9, 1.0):
            bars = swing.load_demo(n_names=250, seed=seed)
            spy = bars.pop("SPY")
            cfg = dict(swing.CFG)
            cfg["tradability_keep_pct"] = keep
            out, _ = swing.run_scan(bars, spy, cfg)
            if len(out):
                found = (bars, spy, cfg, out)
                break
        if found:
            break
    if not found:
        pytest.skip("no candidates in the synthetic universe to veto")

    bars, spy, cfg, baseline = found
    asof = list(bars.values())[0].index[-1].date()

    blackout = {t: (asof + timedelta(days=200), True) for t in bars}
    for ticker in baseline["ticker"]:
        blackout[ticker] = (asof + timedelta(days=2), True)
    vetoed, _ = swing.run_scan(bars, spy, cfg, earnings=blackout)
    assert len(vetoed) < len(baseline)

    clear = {t: (asof + timedelta(days=200), True) for t in bars}
    kept, _ = swing.run_scan(bars, spy, cfg, earnings=clear)
    assert len(kept) == len(baseline), "a distant date must not veto"


def test_the_agent_exposes_every_computing_tool():
    """The tools are the whole point: without them the model can only paraphrase."""
    import ast

    tree = ast.parse(open("chat_agent.py").read())
    registered = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(getattr(d, "id", "") == "tool" for d in node.decorator_list)
    }
    expected = {"check_earnings", "validate_trade_plan", "check_signal_now",
                "get_support_resistance", "size_position", "random_entry_test"}
    assert expected <= registered, sorted(expected - registered)


# ---------------------------------------------------------------------------
# The app's strategies as tools
# ---------------------------------------------------------------------------

def _uptrend_with_pullback(n=320, base=100.0, daily=0.0015, noise=0.0016,
                           pull_len=3, seed=5):
    """The purpose-built §04A fixture: trend, quiet pullback, up-close trigger."""
    import numpy as np

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


@pytest.fixture
def trending(monkeypatch):
    frame = _uptrend_with_pullback()
    bench = data_mod.synthetic_ohlcv(n=len(frame), seed=5, annual_vol=0.16,
                                     annual_drift=0.10)
    bench.index = frame.index
    monkeypatch.setattr(
        at.data_mod, "load_history",
        lambda s, period="3y", **k: bench if s in ("^GSPC", "SPY") else frame)
    return frame


def test_a_firing_setup_comes_back_as_a_complete_bracket(trending):
    info = at.check_swing_setups("TREND")
    assert info["error"] == ""
    assert info["setups"], "the fixture is built to fire §04A"
    for setup in info["setups"]:
        assert setup["stop"] < setup["entry"] <= setup["tp1"] <= setup["tp2"]
        assert setup["r_to_tp2"] > setup["r_to_tp1"] or setup["r_to_tp1"] == 0
        assert setup["order_type"]


def test_the_two_variants_rest_on_opposite_sides_of_the_market(trending):
    for setup in at.check_swing_setups("TREND")["setups"]:
        if setup["variant"].startswith("A"):
            assert "stop-buy above" in setup["order_type"]
        else:
            assert "limit-buy below" in setup["order_type"]


def test_no_setup_is_reported_as_the_normal_answer(loaded):
    """A random walk rarely fires either setup; that must not read as a weak signal."""
    text = at.render_swing_setups(at.check_swing_setups("TEST"))
    if "neither" in text:
        assert "normal answer" in text
        assert "weak signal" in text


def test_a_zero_size_explains_which_rule_zeroed_it(trending):
    """Regression: a risk-off regime zeroes the size, and an unexplained
    "Size 0 shares" reads as a broken number rather than a closed long side."""
    info = at.check_swing_setups("TREND")
    text = at.render_swing_setups(info)
    zero = [s for s in info["setups"] if s["shares"] < 1]
    if zero:
        assert "Size 0" in text
        assert ("regime layer allows no new long risk" in text
                or "stop is too wide" in text)


def test_setups_need_enough_history_for_the_trend_template(monkeypatch):
    short = data_mod.synthetic_ohlcv(n=120, seed=3)
    monkeypatch.setattr(at.data_mod, "load_history", lambda *a, **k: short)
    info = at.check_swing_setups("SHORT")
    assert "252" in info["error"]


def test_the_screen_reports_each_filter_separately(loaded):
    info = at.screen_symbol("TEST")
    assert info["error"] == ""
    assert isinstance(info["liquidity_pass"], bool)
    assert isinstance(info["stage2_trend_template"], bool)
    assert set(info["tradability"]) >= {"adr_pct", "efficiency_ratio", "gap_score",
                                        "vol_regime", "passes_floor"}


def test_a_failed_liquidity_gate_stops_the_screen_there(monkeypatch):
    """Nothing downstream matters if the name cannot be traded at size."""
    thin = data_mod.synthetic_ohlcv(n=400, seed=3)
    thin["Volume"] = 100.0
    monkeypatch.setattr(at.data_mod, "load_history", lambda *a, **k: thin)
    info = at.screen_symbol("THIN")
    assert info["liquidity_pass"] is False
    text = at.render_screen(info)
    assert "FAILS the §01 liquidity gate" in text
    assert "cannot trade it" in text


def test_the_screen_reports_relative_strength_against_the_market(trending):
    info = at.screen_symbol("TREND")
    assert info["rs_126d_vs_market"] is not None
    assert "relative strength" in at.render_screen(info)


def test_the_trend_template_passes_on_a_clean_uptrend(trending):
    assert at.screen_symbol("TREND")["stage2_trend_template"] is True


@pytest.mark.parametrize("strategy", ["breakout", "swing"])
def test_each_strategy_can_be_backtested_by_name(loaded, strategy):
    info = at.backtest_strategy("TEST", strategy=strategy)
    assert info["error"] == ""
    assert info["strategy"] == strategy
    assert "n_trades" in info["metrics"]
    assert "characterisation, not an edge estimate" in at.render_backtest(info)


def test_the_swing_backtest_reports_the_fill_rate(loaded):
    """Orders that never traded are the number that separates this from a
    backtest that assumes every setup becomes a position."""
    info = at.backtest_strategy("TEST", strategy="swing")
    assert {"setups_seen", "orders_placed", "orders_expired", "fill_rate"} \
        <= set(info["metrics"])


def test_the_backtest_compares_against_buy_and_hold(loaded):
    info = at.backtest_strategy("TEST", strategy="breakout")
    assert info["buy_and_hold_return"] is not None
    assert "Buy and hold" in at.render_backtest(info)


def test_an_unknown_strategy_name_is_refused(loaded):
    info = at.backtest_strategy("TEST", strategy="martingale")
    assert "Unknown strategy" in info["error"]


def test_the_scan_lists_what_fired_across_the_watchlist(loaded, monkeypatch, tmp_path):
    import config as app_config
    import watchlist as watchlist_mod

    store = watchlist_mod.WatchlistStore(tmp_path / "watchlist.json")
    for symbol in ("AAA", "BBB"):
        store.add(symbol)
    monkeypatch.setattr(app_config, "WATCHLIST_FILE", tmp_path / "watchlist.json")

    info = at.scan_watchlist("breakout")
    assert info["error"] == ""
    assert info["scanned"] == 2
    assert len(info["hits"]) + len(info["quiet"]) + len(info["failed"]) == 2


def test_an_empty_watchlist_says_so(monkeypatch, tmp_path):
    import config as app_config
    monkeypatch.setattr(app_config, "WATCHLIST_FILE", tmp_path / "empty.json")
    info = at.scan_watchlist("breakout")
    assert "empty" in info["error"]


def test_the_scan_refuses_an_unknown_strategy():
    assert "Unknown strategy" in at.scan_watchlist("astrology")["error"]


def test_the_agent_exposes_every_strategy_tool():
    import ast

    tree = ast.parse(open("chat_agent.py").read())
    registered = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(getattr(d, "id", "") == "tool" for d in node.decorator_list)
    }
    expected = {"check_swing_setups", "screen_symbol", "backtest_strategy",
                "scan_watchlist"}
    assert expected <= registered, sorted(expected - registered)
