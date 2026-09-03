#!/usr/bin/env python3
"""
swing_screener.py — the Swing Universe Funnel, implemented.

Scans a US equity universe once per day, after the close, and prints the
candidates that survive every layer of the spec:

    §01 liquidity gate      hard, binary, unranked
    §02 tradability score   ADR% band, efficiency ratio, gap risk, vol regime
    §03 regime & sector     market-level veto
    §04 setup layer         A: momentum pullback   B: ICT 2022 on daily bars
    §05 event gate          earnings blackout
    §06 book construction   position sizing
    §07 ranking             composite z-score

Long side only. Both setups mirror for shorts; see mirror_note() at the bottom.

Dependencies: pandas, numpy. yfinance only if you use --source yfinance.

    python swing_screener.py --source demo
    python swing_screener.py --source yfinance --tickers AAPL,MSFT,NVDA,...
    python swing_screener.py --source csv --data-dir ./data

CSV format: one <TICKER>.csv per name, columns Date,Open,High,Low,Close,Volume,
split/dividend adjusted, oldest row first.

Not trading advice. Every constant in CFG is a starting value that has not been
validated. Read §08 of the spec before this touches money.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — every threshold in the spec, in one place so §08 can sweep them
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # §01 liquidity gate
    "min_price": 7.00,
    "min_dollar_vol": 25_000_000,      # median over 20 sessions
    "max_pct_of_adv": 0.005,           # your position vs average volume
    "min_history": 90,                 # sessions of listed history

    # §02 tradability
    "adr_min": 1.8,                    # percent
    "adr_max": 7.0,
    "adr_ideal": 3.5,
    "er_min": 0.30,                    # Kaufman efficiency ratio, n=20
    "gap_score_max": 0.15,
    "vol_regime_min": 0.70,            # ATR(14)/ATR(50)
    "vol_regime_max": 1.60,
    "tradability_keep_pct": 0.35,      # keep the top third of gate survivors

    # §03 regime
    "breadth_riskon": 0.50,
    "breadth_riskoff": 0.35,
    "vix_riskoff": 28.0,
    "sector_top_n": 4,

    # §04A momentum pullback
    "rs_percentile_min": 80,
    "pullback_min_len": 3,
    "pullback_max_len": 8,
    "pullback_depth_min": 0.8,         # in ATR
    "pullback_depth_max": 2.5,
    "pullback_ema_touch_atr": 0.5,
    "pullback_vol_ratio": 0.75,
    "pullback_contract_frac": 0.60,
    "stop_buffer_atr": 0.25,
    "tp2_r_multiple": 2.5,
    "max_risk_atr": 2.75,              # safety net; depth_max + buffer sets the real bound
    "min_r_headroom": 2.0,             # room to the 52-week high, unless breaking out
    "breakout_band": 0.98,             # entry this close to the 52w high = no overhead supply

    # §04B ICT on daily bars
    "sweep_window": 3,                 # sessions to reclaim the pool
    "sweep_lookback": 15,              # how far back to hunt for a sweep
    "displacement_atr_k": 1.5,
    "displacement_body_frac": 0.50,
    "equal_level_atr_tol": 0.15,
    "ict_min_r": 2.0,
    "ict_stop_buffer_atr": 0.25,
    "ict_stop_buffer_abs": 0.05,
    "ict_max_risk_atr": 4.0,           # a displacement far above the sweep = unsizeable stop
    "ict_mss_max_age": 5,              # sessions; the daily-bar time stop
    "ict_max_zone_distance_atr": 3.0,  # price must still be able to reach the zone
    "ict_min_r_tp1": 0.5,              # a first target 0.06R away is not a target

    # §05 events
    "earnings_blackout_before": 5,     # sessions
    "earnings_blackout_after": 2,
    "earnings_date_buffer": 3,         # extra days when the date is an estimate

    # §06 book
    "account_equity": 100_000.0,
    "risk_per_trade": 0.0075,          # 0.75%
    "max_positions": 8,

    # §07 ranking weights
    "w_tradability": 0.30,
    "w_setup": 0.30,
    "w_rs": 0.25,
    "w_sector": 0.15,
}


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adr_pct(df: pd.DataFrame, n: int = 20) -> float:
    """§02 average daily range as a percent of close."""
    rng = (df["High"] - df["Low"]) / df["Close"] * 100.0
    return float(rng.tail(n).mean())


def efficiency_ratio(close: pd.Series, n: int = 20) -> float:
    """§02 Kaufman efficiency ratio: net travel over gross travel."""
    seg = close.tail(n + 1)
    if len(seg) < n + 1:
        return np.nan
    net = abs(seg.iloc[-1] - seg.iloc[0])
    gross = seg.diff().abs().sum()
    return float(net / gross) if gross > 0 else 0.0


def gap_score(df: pd.DataFrame, n: int = 60) -> float:
    """§02 fraction of sessions opening more than half an ATR from prior close."""
    a = atr(df, 14).shift(1)
    gap = (df["Open"] - df["Close"].shift(1)).abs() / a
    tail = gap.tail(n).dropna()
    return float((tail > 0.5).mean()) if len(tail) else np.nan


def vol_regime(df: pd.DataFrame) -> float:
    """§02 ATR(14) / ATR(50) — catches collapses and blow-offs."""
    a14, a50 = atr(df, 14).iloc[-1], atr(df, 50).iloc[-1]
    return float(a14 / a50) if a50 and not np.isnan(a50) else np.nan


def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-clip, clip)


def swing_points(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Fractal swings, k=1. A swing at i needs bar i+1, so it is only KNOWN at i+1.
    Both series are shifted by one bar to encode that lag — this is the single
    most important line in the file. Remove the .shift(1) and every backtest
    downstream is fiction (§09, and §06 of the intraday spec).
    """
    h, l = df["High"], df["Low"]
    sh = (h > h.shift(1)) & (h > h.shift(-1))
    sl = (l < l.shift(1)) & (l < l.shift(-1))
    return sh.shift(1).fillna(False), sl.shift(1).fillna(False)


def prior_period_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    §04B liquidity pools: the high and low of the last COMPLETED week and month,
    forward-filled onto each session. Uses only closed periods, never the one in
    progress.
    """
    out = pd.DataFrame(index=df.index)
    for label, rule in (("wk", "W"), ("mo", "ME")):
        grp = df.resample(rule)
        hi, lo = grp["High"].max().shift(1), grp["Low"].min().shift(1)
        out[f"{label}_hi"] = hi.reindex(df.index, method="ffill")
        out[f"{label}_lo"] = lo.reindex(df.index, method="ffill")
    return out


def is_displacement(df: pd.DataFrame, i: int, cfg: dict) -> bool:
    """§04B energetic one-directional delivery: range, body and an FVG."""
    if i < 2:
        return False
    a = atr(df, 20).iloc[i]
    if np.isnan(a) or a == 0:
        return False
    rng = df["High"].iloc[i] - df["Low"].iloc[i]
    body = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
    if rng < cfg["displacement_atr_k"] * a:
        return False
    if rng == 0 or body / rng < cfg["displacement_body_frac"]:
        return False
    return bullish_fvg(df, i) is not None


def bullish_fvg(df: pd.DataFrame, i: int) -> tuple[float, float] | None:
    """Three-bar imbalance: L[i] > H[i-2]. Returns (lower, upper) of the gap."""
    if i < 2:
        return None
    lo, hi = df["High"].iloc[i - 2], df["Low"].iloc[i]
    return (float(lo), float(hi)) if hi > lo else None


# ─────────────────────────────────────────────────────────────────────────────
# §01  Liquidity gate
# ─────────────────────────────────────────────────────────────────────────────

def liquidity_gate(df: pd.DataFrame, cfg: dict, meta: dict | None = None) -> tuple[bool, str]:
    """
    Binary, unranked. Returns (passed, reason_if_failed).

    Market cap, free float and quoted spread are in the spec but not in OHLCV.
    Pass them in `meta` from your data vendor and they are checked; leave them
    out and they are skipped — with the gate weaker than the spec describes.
    """
    if len(df) < cfg["min_history"]:
        return False, f"history {len(df)} < {cfg['min_history']}"

    price = float(df["Close"].iloc[-1])
    if price < cfg["min_price"]:
        return False, f"price {price:.2f}"

    dv = float((df["Close"] * df["Volume"]).tail(20).median())
    if dv < cfg["min_dollar_vol"]:
        return False, f"dollar vol {dv/1e6:.1f}M"

    if meta:
        if meta.get("market_cap") is not None and meta["market_cap"] < 750e6:
            return False, "market cap"
        if meta.get("float_shares") is not None and meta["float_shares"] < 15e6:
            return False, "float"
        if meta.get("spread_bps") is not None and meta["spread_bps"] > 8:
            return False, "spread"
        for flag in ("merger_target", "leveraged_etf", "recent_reverse_split"):
            if meta.get(flag):
                return False, flag

    return True, ""


def max_shares_by_liquidity(df: pd.DataFrame, cfg: dict) -> int:
    """§01 your own size is a filter: cap the position at a slice of ADV."""
    adv = float(df["Volume"].tail(20).median())
    return int(adv * cfg["max_pct_of_adv"])


# ─────────────────────────────────────────────────────────────────────────────
# §02  Tradability
# ─────────────────────────────────────────────────────────────────────────────

def tradability_metrics(df: pd.DataFrame, cfg: dict) -> dict:
    m = {
        "adr_pct": adr_pct(df, 20),
        "er": efficiency_ratio(df["Close"], 20),
        "gap_score": gap_score(df, 60),
        "vol_regime": vol_regime(df),
        "atr14": float(atr(df, 14).iloc[-1]),
        "close": float(df["Close"].iloc[-1]),
    }
    m["passes_floor"] = (
        cfg["adr_min"] <= m["adr_pct"] <= cfg["adr_max"]
        and m["er"] >= cfg["er_min"]
        and m["gap_score"] <= cfg["gap_score_max"]
        and cfg["vol_regime_min"] <= m["vol_regime"] <= cfg["vol_regime_max"]
    )
    return m


def tradability_composite(rows: pd.DataFrame, cfg: dict) -> pd.Series:
    """Cross-sectional z-scores. Only meaningful across the whole universe."""
    adr_fit = -(rows["adr_pct"] - cfg["adr_ideal"]).abs()
    vr_fit = -(rows["vol_regime"] - 1.0).abs()
    return (
        0.35 * zscore(rows["er"])
        + 0.30 * zscore(adr_fit)
        + 0.20 * zscore(-rows["gap_score"])
        + 0.15 * zscore(vr_fit)
    )


# ─────────────────────────────────────────────────────────────────────────────
# §03  Regime & sector
# ─────────────────────────────────────────────────────────────────────────────

def regime_state(spy: pd.DataFrame, breadth: float, vix: float | None, cfg: dict) -> str:
    """Returns 'risk_on' | 'mixed' | 'risk_off'. One state for the whole book."""
    close = spy["Close"]
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    px = float(close.iloc[-1])
    above_50 = px > float(sma50.iloc[-1])
    above_200 = px > float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else True
    rising_50 = float(sma50.iloc[-1]) > float(sma50.iloc[-11])

    if (not above_200) or breadth < cfg["breadth_riskoff"] or (vix and vix > cfg["vix_riskoff"]):
        return "risk_off"
    if above_50 and rising_50 and breadth >= cfg["breadth_riskon"]:
        return "risk_on"
    return "mixed"


def positions_allowed(state: str, cfg: dict) -> int:
    return {"risk_on": cfg["max_positions"], "mixed": 4, "risk_off": 0}[state]


def size_multiplier(state: str) -> float:
    return {"risk_on": 1.0, "mixed": 0.5, "risk_off": 0.0}[state]


def sector_ranks(sector_data: dict[str, pd.DataFrame], cfg: dict) -> dict[str, int]:
    """63-session return of each sector ETF, ranked 1 = strongest."""
    rets = {}
    for etf, d in sector_data.items():
        if len(d) > 63:
            rets[etf] = float(d["Close"].iloc[-1] / d["Close"].iloc[-64] - 1)
    order = sorted(rets, key=rets.get, reverse=True)
    return {etf: i + 1 for i, etf in enumerate(order)}


# ─────────────────────────────────────────────────────────────────────────────
# §04A  Momentum pullback continuation
# ─────────────────────────────────────────────────────────────────────────────

def rs_raw(df: pd.DataFrame) -> float:
    """Recent quarter weighted heavier; the 126-day leg lags 5 bars so the
    pullback you are about to buy is not in the ranking."""
    c = df["Close"]
    if len(c) < 132:
        return np.nan
    r63 = c.iloc[-1] / c.iloc[-64] - 1
    r126 = c.iloc[-6] / c.iloc[-132] - 1
    return float(0.6 * r63 + 0.4 * r126)


def trend_template(df: pd.DataFrame) -> bool:
    """All five conditions, no partial credit."""
    c = df["Close"]
    if len(c) < 252:
        return False
    s50, s150, s200 = (c.rolling(n).mean() for n in (50, 150, 200))
    px = float(c.iloc[-1])
    return bool(
        px > s50.iloc[-1] > s150.iloc[-1] > s200.iloc[-1]
        and s200.iloc[-1] > s200.iloc[-21]
        and px >= 0.75 * float(df["High"].tail(252).max())
        and px >= 1.30 * float(df["Low"].tail(252).min())
    )


def momentum_setup(df: pd.DataFrame, cfg: dict) -> dict | None:
    """Evaluated at the last completed bar. Entry is a stop-buy for tomorrow."""
    if not trend_template(df):
        return None

    a = float(atr(df, 14).iloc[-1])
    if np.isnan(a) or a <= 0:
        return None

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    sma50 = c.rolling(50).mean()

    # The trigger bar must be an up-close.
    if c.iloc[-1] <= c.iloc[-2]:
        return None

    # Walk back over legal pullback lengths and take the first that qualifies.
    for length in range(cfg["pullback_min_len"], cfg["pullback_max_len"] + 1):
        seg = df.iloc[-(length + 1):-1]          # the pullback, excluding today
        if len(seg) < length:
            break

        swing_hi = float(df["High"].iloc[-(length + 6):-(length)].max())
        low = float(seg["Low"].min())
        depth = (swing_hi - low) / a
        if not (cfg["pullback_depth_min"] <= depth <= cfg["pullback_depth_max"]):
            continue

        # Must have leaned on a short EMA, and must not have closed under the 50.
        near = min(abs(low - float(ema10.iloc[-2])), abs(low - float(ema20.iloc[-2])))
        if near > cfg["pullback_ema_touch_atr"] * a:
            continue
        if (seg["Close"] < sma50.reindex(seg.index)).any():
            continue

        # Volume drying up, ranges contracting: a pause, not distribution.
        if float(seg["Volume"].mean()) > cfg["pullback_vol_ratio"] * float(v.tail(20).mean()):
            continue
        rng = seg["High"] - seg["Low"]
        if len(rng) > 1:
            contracting = (rng.diff().dropna() <= 0).mean()
            if contracting < cfg["pullback_contract_frac"]:
                continue

        entry = float(h.iloc[-1]) + 0.05
        stop = low - cfg["stop_buffer_atr"] * a
        risk = entry - stop
        if risk <= 0 or risk > cfg["max_risk_atr"] * a:
            continue

        # TP1 is the prior swing high — usually LESS than 1R away, which is the
        # point: it is where you take a partial and move the stop, not where the
        # trade is supposed to pay. The R test belongs on the headroom above it.
        tp1 = swing_hi if swing_hi > entry + 0.25 * risk else entry + risk

        # Room to run. A name entering at its 52-week high has no overhead
        # supply and passes by definition; one sitting just under a big level
        # with less than 2R of space does not.
        high252 = float(h.tail(252).max())
        if entry < cfg["breakout_band"] * high252:
            if (high252 - entry) / risk < cfg["min_r_headroom"]:
                continue

        return {
            "variant": "A/momentum",
            "entry": entry,
            "stop": stop,
            "tp1": tp1,
            "tp2": entry + cfg["tp2_r_multiple"] * risk,
            "r_to_tp1": (tp1 - entry) / risk,
            # Setup quality: tighter pullback and stronger volume dry-up score better.
            "quality": float((1.0 / max(depth, 0.1)) + (1.0 - float(seg["Volume"].mean()) / max(float(v.tail(20).mean()), 1))),
            "note": f"pullback {length}d, depth {depth:.2f} ATR",
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# §04B  The 2022 model on daily bars
# ─────────────────────────────────────────────────────────────────────────────

def ict_setup(df: pd.DataFrame, cfg: dict) -> dict | None:
    """
    sweep sell-side → displacement MSS through the last confirmed swing high →
    entry in the FVG the displacement left → draw to the opposing pool.

    Long side. The gap variant of the sweep is included: in stocks the raid is
    often an overnight gap through the pool that gets reclaimed intraday.
    """
    n = len(df)
    if n < 120:
        return None

    a14 = atr(df, 14)
    a = float(a14.iloc[-1])
    if np.isnan(a) or a <= 0:
        return None

    lv = prior_period_levels(df)
    sh, _ = swing_points(df)                       # already lagged by one bar
    t = n - 1                                      # last completed bar

    # 1 ── find the most recent sweep of a sell-side pool
    sweep = None
    for i in range(max(2, t - cfg["sweep_lookback"]), t + 1):
        for pool_name in ("wk_lo", "mo_lo"):
            pool = lv[pool_name].iloc[i]
            if np.isnan(pool):
                continue
            low_i, open_i, close_i = (float(df[c].iloc[i]) for c in ("Low", "Open", "Close"))
            wick_raid = low_i < pool
            gap_raid = open_i < pool                     # stock-specific variant
            if not (wick_raid or gap_raid):
                continue
            # reclaimed within N sessions?
            for j in range(i, min(i + cfg["sweep_window"], t) + 1):
                if float(df["Close"].iloc[j]) > pool:
                    sweep = {
                        "i": i, "j": j, "pool": float(pool), "pool_name": pool_name,
                        "extreme": float(df["Low"].iloc[i:j + 1].min()),
                        "kind": "gap" if gap_raid and not wick_raid else "wick",
                    }
                    break
            if sweep:
                break
        if sweep:
            break
    if sweep is None:
        return None

    # 2 ── market structure shift: displacement close above the last confirmed
    #      swing high formed since the sweep
    mss = None
    for k in range(sweep["j"], t + 1):
        highs = [float(df["High"].iloc[x]) for x in range(sweep["i"], k) if sh.iloc[x]]
        if not highs:
            continue
        ref = max(highs)
        if float(df["Close"].iloc[k]) > ref and is_displacement(df, k, cfg):
            mss = {"k": k, "ref": ref}
            break
    if mss is None:
        return None

    # invalidated if price closed back below the manipulation extreme
    if float(df["Close"].iloc[mss["k"]:t + 1].min()) < sweep["extreme"]:
        return None

    # time stop: a sequence that shifted structure two weeks ago is not a live
    # setup, it is history. The intraday spec expires ARMED and PENDING states;
    # this is the daily-bar equivalent.
    if t - mss["k"] > cfg["ict_mss_max_age"]:
        return None

    # 3 ── entry zone: the FVG inside the displacement leg
    fvg = bullish_fvg(df, mss["k"])
    if fvg is None:
        return None
    gap_lo, gap_hi = fvg
    entry = gap_hi                                   # proximal edge, highest fill rate
    stop = sweep["extreme"] - max(cfg["ict_stop_buffer_atr"] * a, cfg["ict_stop_buffer_abs"])
    risk = entry - stop
    if risk <= 0 or risk > cfg["ict_max_risk_atr"] * a:
        return None

    # The zone must still be reachable. Below it, the setup already failed;
    # far above it, price has run away and a limit order down there is not a
    # live order, it is a wish.
    px = float(df["Close"].iloc[t])
    if px < gap_lo or px > entry + cfg["ict_max_zone_distance_atr"] * a:
        return None

    # 4 ── draw on liquidity: nearest untapped opposing pool at least min_r away.
    #      Internal liquidity (the last confirmed swing high) counts as TP1 —
    #      without it TP1 collapses onto the 52-week high and reads as a 9R
    #      first target, which no swing trade actually has.
    internal = [float(df["High"].iloc[x]) for x in range(max(0, t - 60), t + 1)
                if sh.iloc[x] and float(df["High"].iloc[x]) > entry]
    targets = sorted(
        [p for p in (lv["wk_hi"].iloc[t], lv["mo_hi"].iloc[t],
                     float(df["High"].tail(252).max()))
         if not np.isnan(p) and p > entry] + internal
    )
    if not targets:
        return None
    dol = next((p for p in targets if (p - entry) / risk >= cfg["ict_min_r"]), None)
    if dol is None:
        return None
    tp1 = next((p for p in targets if (p - entry) / risk >= cfg["ict_min_r_tp1"]), dol)

    return {
        "variant": "B/ict",
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": dol,
        "r_to_tp1": (tp1 - entry) / risk,
        # Setup quality: displacement strength, in ATR.
        "quality": float((df["High"].iloc[mss["k"]] - df["Low"].iloc[mss["k"]]) / a),
        "note": f"{sweep['kind']} sweep of {sweep['pool_name']} @ {sweep['pool']:.2f}, "
                f"MSS +{t - mss['k']}d ago",
    }


# ─────────────────────────────────────────────────────────────────────────────
# §05  Event gate
# ─────────────────────────────────────────────────────────────────────────────

def earnings_veto(ticker: str, asof: date, earnings: dict, cfg: dict) -> str:
    """
    `earnings` maps ticker -> (date, confirmed: bool). An estimated date is
    treated as real and buffered on both sides — vendors are routinely wrong by
    a week, and this is the most common way a well-selected swing trade dies.
    """
    ent = earnings.get(ticker)
    if not ent:
        return ""
    ed, confirmed = ent
    buf = 0 if confirmed else cfg["earnings_date_buffer"]
    # Calendar days as a conservative stand-in for sessions.
    before = cfg["earnings_blackout_before"] * 7 / 5 + buf
    after = cfg["earnings_blackout_after"] * 7 / 5 + buf
    delta = (ed - asof).days
    if -after <= delta <= before:
        return f"earnings {ed:%Y-%m-%d}{'' if confirmed else ' (est)'}"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# §06  Sizing
# ─────────────────────────────────────────────────────────────────────────────

def position_size(entry: float, stop: float, df: pd.DataFrame, state: str, cfg: dict) -> int:
    """Shares follow from the stop, never the reverse. Then capped by ADV."""
    risk_dollars = cfg["account_equity"] * cfg["risk_per_trade"] * size_multiplier(state)
    per_share = entry - stop
    if per_share <= 0 or risk_dollars <= 0:
        return 0
    return max(0, min(int(risk_dollars / per_share), max_shares_by_liquidity(df, cfg)))


# ─────────────────────────────────────────────────────────────────────────────
# The scan
# ─────────────────────────────────────────────────────────────────────────────

def run_scan(bars: dict[str, pd.DataFrame], spy: pd.DataFrame, cfg: dict,
             vix: float | None = None, earnings: dict | None = None,
             sector_map: dict | None = None, sector_data: dict | None = None,
             meta: dict | None = None) -> tuple[pd.DataFrame, dict]:
    earnings = earnings or {}
    meta = meta or {}
    asof = bars[next(iter(bars))].index[-1].date()

    # §01 ─────────────────────────────────────────────────────────────────────
    gated, rejects = {}, {}
    for tkr, df in bars.items():
        ok, why = liquidity_gate(df, cfg, meta.get(tkr))
        (gated.__setitem__(tkr, df) if ok else rejects.__setitem__(tkr, why))

    if not gated:
        # Callers read regime/breadth/slots off ctx unconditionally, so this
        # early return has to carry them too.
        return pd.DataFrame(), {
            "asof": asof, "stage": "gate", "universe": len(bars), "gated": 0,
            "floor_ok": 0, "tradable": 0, "breadth": float("nan"),
            "regime": "unknown", "slots": 0, "rejects": rejects,
        }

    # §02 ─────────────────────────────────────────────────────────────────────
    rows = pd.DataFrame({t: tradability_metrics(d, cfg) for t, d in gated.items()}).T
    rows = rows.astype({c: float for c in
                        ("adr_pct", "er", "gap_score", "vol_regime", "atr14", "close")})
    rows["tradability"] = tradability_composite(rows, cfg)
    floor_ok = rows[rows["passes_floor"].astype(bool)]
    keep = max(1, int(len(floor_ok) * cfg["tradability_keep_pct"]))
    tradable = floor_ok.nlargest(keep, "tradability")

    # §03 ─────────────────────────────────────────────────────────────────────
    breadth = float(np.mean([
        float(d["Close"].iloc[-1]) > float(d["Close"].rolling(50).mean().iloc[-1])
        for d in gated.values() if len(d) >= 50
    ]))
    state = regime_state(spy, breadth, vix, cfg)
    sranks = sector_ranks(sector_data, cfg) if sector_data else {}

    ctx = {"asof": asof, "universe": len(bars), "gated": len(gated),
           "floor_ok": len(floor_ok), "tradable": len(tradable),
           "breadth": breadth, "regime": state,
           "slots": positions_allowed(state, cfg), "rejects": rejects}

    if state == "risk_off":
        ctx["stage"] = "regime veto — longs off"
        return pd.DataFrame(), ctx

    # §04 / §05 ───────────────────────────────────────────────────────────────
    cands = []
    for tkr in tradable.index:
        df = gated[tkr]

        if sector_map and sranks:
            rank = sranks.get(sector_map.get(tkr))
            if rank is not None and rank > cfg["sector_top_n"]:
                continue

        veto = earnings_veto(tkr, asof, earnings, cfg)
        if veto:
            continue

        for setup in (momentum_setup(df, cfg), ict_setup(df, cfg)):
            if not setup:
                continue
            shares = position_size(setup["entry"], setup["stop"], df, state, cfg)
            if shares < 1:
                continue
            cands.append({
                "ticker": tkr,
                "variant": setup["variant"],
                "close": float(df["Close"].iloc[-1]),
                "entry": round(setup["entry"], 2),
                "stop": round(setup["stop"], 2),
                "tp1": round(setup["tp1"], 2),
                "tp2": round(setup["tp2"], 2),
                "r_tp1": round(setup["r_to_tp1"], 2),
                "shares": shares,
                "risk_$": round(shares * (setup["entry"] - setup["stop"]), 0),
                "adr%": round(float(tradable.loc[tkr, "adr_pct"]), 2),
                "er": round(float(tradable.loc[tkr, "er"]), 2),
                "gap": round(float(tradable.loc[tkr, "gap_score"]), 2),
                "tradability": float(tradable.loc[tkr, "tradability"]),
                "quality": float(setup["quality"]),
                "rs_raw": rs_raw(df),
                "sector_rank": sranks.get((sector_map or {}).get(tkr), np.nan),
                "note": setup["note"],
            })

    if not cands:
        ctx["stage"] = "no setup today"
        return pd.DataFrame(), ctx

    # §07 ─────────────────────────────────────────────────────────────────────
    out = pd.DataFrame(cands)
    out["rs_pct"] = out["rs_raw"].rank(pct=True) * 100
    sector_term = zscore(-out["sector_rank"].fillna(out["sector_rank"].mean() if out["sector_rank"].notna().any() else 0))
    out["score"] = (
        cfg["w_tradability"] * zscore(out["tradability"])
        + cfg["w_setup"] * zscore(out["quality"])
        + cfg["w_rs"] * zscore(out["rs_pct"])
        + cfg["w_sector"] * sector_term
    )
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    ctx["stage"] = "ok"
    ctx["candidates"] = len(out)
    return out, ctx


# ─────────────────────────────────────────────────────────────────────────────
# Data sources
# ─────────────────────────────────────────────────────────────────────────────

def load_yfinance(tickers: list[str], years: int = 3) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    raw = yf.download(tickers, period=f"{years}y", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d.dropna()[["Open", "High", "Low", "Close", "Volume"]]
            if len(d) > 250:
                out[t] = d
        except (KeyError, TypeError):
            pass
    return out


def load_csv(data_dir: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        t = os.path.splitext(os.path.basename(path))[0].upper()
        d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        out[t] = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return out


def load_demo(n_names: int = 40, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Synthetic bars so the pipeline can be run and read without a data feed."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
    out = {}
    for k in range(n_names):
        tkr = f"SYN{k:02d}"
        drift = rng.normal(0.0006, 0.0006)
        vol = rng.uniform(0.012, 0.035)
        r = rng.normal(drift, vol, len(idx))
        # give a third of the names a real trend so setups can actually fire
        if k % 3 == 0:
            r += np.linspace(0, 0.0012, len(idx))
        close = 40 * np.exp(np.cumsum(r))
        prev = np.concatenate([[close[0]], close[:-1]])
        # opens sit near the prior close, as real ones do — otherwise the §02
        # gap-risk filter rejects the entire synthetic universe
        openp = prev * (1 + rng.normal(0, 0.25 * vol, len(idx)))
        wick = close * rng.uniform(0.002, 0.010, len(idx))
        high = np.maximum(openp, close) + wick * rng.uniform(0.2, 1.0, len(idx))
        low = np.minimum(openp, close) - wick * rng.uniform(0.2, 1.0, len(idx))
        # Volume has to track price activity. Drawn independently, a pullback
        # segment averages the same volume as the trailing 20 days, so the
        # §04A "volume dries up" test can never pass and the momentum setup
        # never fires on demo data.
        activity = np.abs(r) / vol
        vols = rng.lognormal(math.log(3e6), 0.25, len(idx)) * (0.55 + 0.75 * activity)
        out[tkr] = pd.DataFrame(
            {"Open": openp, "High": high, "Low": low,
             "Close": close, "Volume": vols}, index=idx)
    # a market proxy built from the same process
    mkt = sum(d["Close"] / d["Close"].iloc[0] for d in out.values()) / len(out) * 400
    out["SPY"] = pd.DataFrame(
        {"Open": mkt, "High": mkt * 1.004, "Low": mkt * 0.996,
         "Close": mkt, "Volume": 8e7}, index=idx)
    return out


# ─────────────────────────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Swing Universe Funnel screener")
    ap.add_argument("--source", choices=("demo", "yfinance", "csv"), default="demo")
    ap.add_argument("--tickers", help="comma-separated, for --source yfinance")
    ap.add_argument("--data-dir", default="./data", help="for --source csv")
    ap.add_argument("--equity", type=float, default=CFG["account_equity"])
    ap.add_argument("--out", default="candidates.csv")
    args = ap.parse_args()

    CFG["account_equity"] = args.equity
    sector_data = None

    if args.source == "demo":
        bars = load_demo()
    elif args.source == "csv":
        bars = load_csv(args.data_dir)
    else:
        if not args.tickers:
            print("--tickers required for --source yfinance", file=sys.stderr)
            return 2
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        bars = load_yfinance(sorted(set(tickers + ["SPY"])))
        sector_data = load_yfinance(SECTOR_ETFS)

    if "SPY" not in bars:
        print("no SPY series — the §03 regime layer needs one", file=sys.stderr)
        return 2
    spy = bars.pop("SPY")

    out, ctx = run_scan(bars, spy, CFG, sector_data=sector_data)

    print(f"\n  as of {ctx['asof']}   regime: {ctx['regime'].upper()}   "
          f"breadth {ctx['breadth']:.0%}   slots {ctx['slots']}")
    print(f"  funnel: {ctx['universe']} universe → {ctx['gated']} gate → "
          f"{ctx['floor_ok']} floor → {ctx['tradable']} tradable → "
          f"{len(out)} with a setup\n")

    if out.empty:
        print(f"  no candidates — {ctx['stage']}\n")
        return 0

    cols = ["ticker", "variant", "close", "entry", "stop", "tp1", "tp2",
            "r_tp1", "shares", "risk_$", "adr%", "er", "gap", "score"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out[cols].head(10).to_string(index=False,
              formatters={"score": "{:.2f}".format}))
    print()
    for _, r in out.head(ctx["slots"]).iterrows():
        print(f"  {r['ticker']:<8} {r['variant']:<12} {r['note']}")
    out.to_csv(args.out, index=False)
    print(f"\n  wrote {args.out}  ({len(out)} rows)\n")
    return 0


def mirror_note() -> str:
    """
    Short side, for when you want it:
      §04A  invert the trend template (C < 50 < 150 < 200DMA, falling), rank on
            the BOTTOM RS percentile, look for a low-volume RALLY into the EMAs,
            trigger on a down-close, stop above the rally high.
      §04B  bearish_fvg is H[i] < L[i-2]; sweep the wk_hi/mo_hi pools; MSS is a
            displacement close BELOW the last confirmed swing low.
      §01   add the borrow check — hard-to-borrow means recall risk mid-swing.
      §03   shorts are half size in 'mixed' and the only side open in 'risk_off'.
    """
    return mirror_note.__doc__


if __name__ == "__main__":
    raise SystemExit(main())
