"""Monthly-rebalanced max-Sharpe portfolio over the screener's top Nasdaq-100 names.

The engine behind ``notebooks/qqq_top20_max_sharpe_backtest.ipynb`` (which runs it)
and ``notebooks/qqq_backtest_visualization.ipynb`` (which draws it). It lives in a
module for the same reason ``strategy.py`` does: two notebooks holding two copies of
the rules is two strategies, and the second one diverges silently.

Selection is not re-implemented here. ``screen_asof`` calls
``screening.evaluate_ticker`` — the Screener tab's own function — on price history
truncated at the as-of date. That function reads ``.iloc[-1]``, so it always answers
"as of the last bar you hand it"; truncating the input is what makes it
point-in-time. Nothing downstream of the truncation can see a later bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import data as data_mod
import screening
from backtest import TRADING_DAYS, CostModel
from config import SCREENING_PARAMS
from strategy import Action

# The screen resolves each name's sector through yfinance's `.info` endpoint. Sector
# is a *label* on the result — `passed` and `daily_annret`, the only two fields the
# ranking reads, never touch it — so it is stubbed rather than firing one network
# call per name per rebalance.
screening.yf_info_cached = lambda ticker: {}


@dataclass(frozen=True)
class BacktestSettings:
    principal: float = 10_000.0
    start: str = "2025-06-01"
    end: Optional[str] = None          # None = today

    top_n: int = 20
    benchmarks: tuple[str, ...] = ("QQQ", "BOXX")
    #: The screen's relative-strength benchmark. Same as the dashboard's.
    rs_benchmark: str = SCREENING_PARAMS["market_benchmark"]
    #: Trailing window handed to the screen, in bars. 3y matches SCREENING_PARAMS["period"].
    screen_window: int = 756

    #: Trailing daily returns used to estimate the covariance for the optimiser.
    cov_lookback: int = 252
    #: Below this many overlapping bars a name cannot be optimised and is dropped.
    cov_min_bars: int = 60
    #: Shrinkage of the sample covariance toward its diagonal. 0 = raw sample.
    cov_shrinkage: float = 0.10
    #: Hard cap on one name's weight. Mirrors config.MAX_POSITION_PCT.
    max_weight: float = 0.25
    #: Symbol whose trailing return stands in for the risk-free rate, or None for 0.
    rf_proxy: Optional[str] = "BOXX"

    costs: CostModel = field(default_factory=CostModel)
    #: How much of each rebalance's equity is held back to cover slippage and commission.
    cost_buffer: float = 0.002

    #: History pulled per symbol. Must cover screen_window before `start`.
    history_period: str = "5y"

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end) if self.end else pd.Timestamp.today().normalize()


# ── Universe ────────────────────────────────────────────────────────────────────

#: Snapshot fallback for an offline or rate-limited run. The live scrape is preferred;
#: holding this fixed across a window is where the notebooks' membership bias comes from.
NDX_STATIC = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC",
    "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP",
    "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR",
    "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA",
    "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]


def nasdaq100_symbols(verbose: bool = True) -> list[str]:
    """Current Nasdaq-100 tickers, scraped, falling back to the snapshot above."""
    try:
        import requests
        from bs4 import BeautifulSoup

        res = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        header = [th.get_text(strip=True).lower() for th in table.find("tr").find_all(["th", "td"])]
        col = header.index("ticker") if "ticker" in header else header.index("symbol")
        out = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) > col:
                out.append(cells[col].get_text(strip=True).replace(".", "-"))
        if len(out) >= 90:
            if verbose:
                print(f"universe: scraped {len(out)} Nasdaq-100 tickers from Wikipedia")
            return sorted(set(out))
    except Exception as exc:
        if verbose:
            print(f"universe: scrape failed ({type(exc).__name__}: {exc})")

    if verbose:
        print(f"universe: using the static snapshot ({len(NDX_STATIC)} tickers)")
    return sorted(set(NDX_STATIC))


# ── Price history ───────────────────────────────────────────────────────────────

@dataclass
class MarketData:
    """Everything the back-test reads, aligned onto one trading calendar.

    ``prices`` keeps each symbol's raw frame — that is what the screen is handed, so
    its indicators are computed on real bars. ``closes`` and ``opens`` are the same
    data reindexed onto ``master`` and forward-filled: a symbol missing one bar then
    carries its last close instead of dropping out of the mark-to-market and putting
    a hole in the equity curve.
    """

    prices: dict[str, pd.DataFrame]
    closes: pd.DataFrame
    opens: pd.DataFrame
    master: pd.DatetimeIndex
    calendar: pd.DatetimeIndex
    synthetic: bool


def load_real_prices(symbols: Sequence[str], cfg: BacktestSettings,
                     verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Per-symbol OHLCV through the repo's cached loader. Missing symbols are omitted."""
    frames: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        frame = data_mod.load_history(sym, period=cfg.history_period)
        if not frame.empty:
            frames[sym] = frame
        if verbose and i % 25 == 0:
            print(f"  loaded {i}/{len(symbols)} ({len(frames)} with data)")
    return frames


def synthetic_prices(symbols: Sequence[str], cfg: BacktestSettings,
                     synth_start: str = "2020-10-01") -> dict[str, pd.DataFrame]:
    """Deterministic stand-in history, one distinct path per symbol.

    The same escape hatch ``backtest.py --demo`` uses. A run on this data exercises
    the machinery and says nothing whatsoever about the strategy.
    """
    n = len(pd.bdate_range(synth_start, cfg.end_ts))
    profile = {"BOXX": (0.05, 0.006), "QQQ": (0.13, 0.19), cfg.rs_benchmark: (0.10, 0.15)}
    frames = {}
    for seed, sym in enumerate(symbols):
        drift, vol = profile.get(sym, (0.04 + 0.16 * ((seed % 7) / 6.0), 0.20 + 0.05 * (seed % 5)))
        frames[sym] = data_mod.synthetic_ohlcv(
            n=n, seed=1000 + seed, start=synth_start,
            initial_price=40.0 + 3.0 * (seed % 40),
            annual_drift=drift, annual_vol=vol,
        )
    return frames


def load_market(universe: Sequence[str], cfg: BacktestSettings,
                verbose: bool = True) -> MarketData:
    """Load every symbol the back-test needs, falling back to synthetic history."""
    needed = list(dict.fromkeys([*universe, *cfg.benchmarks, cfg.rs_benchmark]))
    if verbose:
        print(f"loading {len(needed)} symbols...")

    prices = load_real_prices(needed, cfg, verbose=verbose)
    required = {*cfg.benchmarks, cfg.rs_benchmark}
    synthetic = len(prices) < len(needed) // 2 or any(s not in prices for s in required)

    if synthetic:
        if verbose:
            print("\n" + "!" * 78)
            print("!! SYNTHETIC MODE — real prices are unavailable in this environment.")
            print("!! Every number below is generated from a random walk. It is a test of the")
            print("!! machinery, not a result. Re-run where yfinance is reachable for real figures.")
            print("!" * 78 + "\n")
        prices = synthetic_prices(needed, cfg)

    # QQQ is the exchange calendar the strategy actually trades on; a union across 100
    # symbols would pick up stray dates from a single mis-stamped series.
    master = (prices["QQQ"].index if "QQQ" in prices
              else pd.DatetimeIndex(sorted(set().union(*[f.index for f in prices.values()]))))
    closes = pd.DataFrame({s: f["Close"] for s, f in prices.items()}).reindex(master).ffill()
    opens = pd.DataFrame({s: f["Open"] for s, f in prices.items()}).reindex(master).ffill()
    calendar = master[(master >= cfg.start_ts) & (master <= cfg.end_ts)]

    if verbose:
        print(f"mode: {'SYNTHETIC' if synthetic else 'live prices'} | symbols with data: {len(prices)}")
        print(f"history: {closes.index[0].date()} -> {closes.index[-1].date()}")
        print(f"backtest window: {calendar[0].date()} -> {calendar[-1].date()} "
              f"({len(calendar)} trading days)")

    return MarketData(prices, closes, opens, master, calendar, synthetic)


# ── Selection: the dashboard's screen, asked point-in-time ──────────────────────

def screen_asof(market: MarketData, asof: pd.Timestamp, universe: Iterable[str],
                cfg: BacktestSettings, passed_only: bool = True) -> pd.DataFrame:
    """Run the dashboard's screen using only bars at or before ``asof``.

    With ``passed_only=False`` every name the screen could evaluate comes back, with
    its ``passed`` flag intact — which is what a funnel or a scatter of the rejected
    names needs.
    """
    bench = market.closes[cfg.rs_benchmark].loc[:asof].dropna()
    if len(bench) < SCREENING_PARAMS["rs_lookback"]:
        return pd.DataFrame()

    rows = []
    for sym in universe:
        frame = market.prices.get(sym)
        if frame is None:
            continue
        window = frame.loc[:asof].tail(cfg.screen_window)
        if len(window) < SCREENING_PARAMS["rs_lookback"]:
            continue
        try:
            res = screening.evaluate_ticker(sym, window, bench)
        except Exception:
            continue
        if res is not None:
            rows.append(res.__dict__)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if passed_only:
        df = df[df["passed"]]
    return df.sort_values("daily_annret", ascending=False).reset_index(drop=True)


def select_top_n(market: MarketData, asof: pd.Timestamp, universe: Iterable[str],
                 cfg: BacktestSettings) -> list[str]:
    """The screen's top ``cfg.top_n`` tickers by annualised return, as of ``asof``."""
    passed = screen_asof(market, asof, universe, cfg)
    return [] if passed.empty else passed["ticker"].head(cfg.top_n).tolist()


# ── Sizing: long-only maximum Sharpe ────────────────────────────────────────────

def annualised_moments(rets: pd.DataFrame, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    """Annualised mean vector and shrunk covariance from daily returns.

    A 252-by-20 sample covariance is badly conditioned; inverting it raw concentrates
    the whole book in whichever name happened to be quietest. Shrinkage pulls the
    off-diagonals toward zero.
    """
    mu = rets.mean().to_numpy() * TRADING_DAYS
    sample = rets.cov().to_numpy() * TRADING_DAYS
    cov = (1.0 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
    cov = (cov + cov.T) / 2.0                 # a sample covariance can come back
    cov += np.eye(len(cov)) * 1e-10           # marginally indefinite; nudge onto the PSD cone
    return mu, cov


def _solve(objective, n: int, max_weight: float, starts: list[np.ndarray]):
    bounds = [(0.0, max_weight)] * n
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    best, best_val = None, np.inf
    for w0 in starts:
        try:
            res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"maxiter": 400, "ftol": 1e-10})
        except Exception:
            continue
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best, best_val = res.x, res.fun
    return best


def _starts(n: int, cov: np.ndarray, max_weight: float, seed: int) -> list[np.ndarray]:
    """Equal weight, inverse volatility, and random draws.

    SLSQP on the Sharpe objective is not convex and will happily stop at a local
    optimum, so it is started from several places and the best answer kept.
    """
    vols = np.sqrt(np.diag(cov))
    inv_vol = np.nan_to_num(1.0 / np.where(vols > 0, vols, np.nan), nan=0.0)
    inv_vol = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else np.full(n, 1.0 / n)

    rng = np.random.default_rng(seed)
    out = [np.full(n, 1.0 / n), np.clip(inv_vol, 0.0, max_weight)]
    out = [s / s.sum() for s in out]
    for _ in range(8):
        draw = np.clip(rng.dirichlet(np.ones(n)), 0.0, max_weight)
        out.append(draw / draw.sum())
    return out


def max_sharpe_weights(rets: pd.DataFrame, rf: float, cfg: BacktestSettings,
                       seed: int = 0) -> tuple[pd.Series, str]:
    """Long-only tangency weights, and a note on how they were reached.

    Falls back to minimum variance when no name has a trailing excess return above
    ``rf``: the Sharpe objective has no meaningful maximum there, and returning
    whatever the optimiser lands on would be worse than saying so.
    """
    n = rets.shape[1]
    if n == 0:
        return pd.Series(dtype=float), "empty"
    if n == 1:
        return pd.Series([1.0], index=rets.columns), "single name"

    max_weight = max(cfg.max_weight, 1.0 / n)   # keep the feasible set non-empty
    mu, cov = annualised_moments(rets, cfg.cov_shrinkage)
    starts = _starts(n, cov, max_weight, seed)

    def neg_variance(w):
        return float(w @ cov @ w)

    if mu.max() <= rf:
        w, note = _solve(neg_variance, n, max_weight, starts), "min-variance fallback (no name above rf)"
    else:
        def neg_sharpe(w):
            vol = np.sqrt(max(float(w @ cov @ w), 1e-18))
            return -(float(w @ mu) - rf) / vol

        w, note = _solve(neg_sharpe, n, max_weight, starts), "max-sharpe"
        if w is None:
            w, note = _solve(neg_variance, n, max_weight, starts), "min-variance fallback (optimiser failed)"

    if w is None:
        return pd.Series(np.full(n, 1.0 / n), index=rets.columns), "equal weight fallback"

    w = np.clip(w, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    return pd.Series(w, index=rets.columns), note


def min_variance_weights(rets: pd.DataFrame, cfg: BacktestSettings, seed: int = 0) -> pd.Series:
    """Minimum-variance weights under the same constraints. For the frontier plot."""
    n = rets.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(cfg.max_weight, 1.0 / n)
    _, cov = annualised_moments(rets, cfg.cov_shrinkage)
    w = _solve(lambda x: float(x @ cov @ x), n, max_weight, _starts(n, cov, max_weight, seed))
    if w is None:
        w = np.full(n, 1.0 / n)
    return pd.Series(w / w.sum(), index=rets.columns)


def min_variance_for_return(rets: pd.DataFrame, target: float, cfg: BacktestSettings):
    """Lowest-variance weights reaching ``target`` annualised return, or None.

    One point on the efficient frontier. Returns None when the target is outside what
    the constraint set can reach, which is how the frontier finds its own endpoints.
    """
    n = rets.shape[1]
    if n == 0:
        return None
    max_weight = max(cfg.max_weight, 1.0 / n)
    mu, cov = annualised_moments(rets, cfg.cov_shrinkage)
    bounds = [(0.0, max_weight)] * n
    constraints = (
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target},
    )
    for w0 in _starts(n, cov, max_weight, seed=1):
        try:
            res = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"maxiter": 400, "ftol": 1e-10})
        except Exception:
            continue
        if res.success and abs(float(res.x @ mu) - target) < 1e-4:
            return pd.Series(res.x, index=rets.columns)
    return None


def portfolio_return(weights, mu) -> float:
    return float(np.asarray(weights) @ np.asarray(mu))


def portfolio_vol(weights, cov) -> float:
    w = np.asarray(weights)
    return float(np.sqrt(max(w @ np.asarray(cov) @ w, 0.0)))


def risk_free_rate(market: MarketData, asof: pd.Timestamp, cfg: BacktestSettings) -> float:
    """Annualised trailing return of the risk-free proxy, or 0.0 without one."""
    if not cfg.rf_proxy or cfg.rf_proxy not in market.closes.columns:
        return 0.0
    series = market.closes[cfg.rf_proxy].loc[:asof].dropna().tail(cfg.cov_lookback)
    if len(series) < cfg.cov_min_bars:
        return 0.0
    return float(series.pct_change().dropna().mean() * TRADING_DAYS)


def selected_returns(market: MarketData, asof: pd.Timestamp, picks: Sequence[str],
                     cfg: BacktestSettings) -> pd.DataFrame:
    """Trailing daily returns for ``picks``, as the optimiser sees them at ``asof``."""
    if not picks:
        return pd.DataFrame()
    window = market.closes[list(picks)].loc[:asof].tail(cfg.cov_lookback + 1)
    rets = window.pct_change().dropna(how="all")
    return rets.dropna(axis=1, thresh=cfg.cov_min_bars).dropna()


def target_weights(market: MarketData, asof: pd.Timestamp, universe: Iterable[str],
                   cfg: BacktestSettings) -> tuple[pd.Series, list[str], str]:
    """(weights, picks, note) for the month beginning after ``asof``."""
    picks = select_top_n(market, asof, universe, cfg)
    if not picks:
        return pd.Series(dtype=float), [], "no name passed the screen — hold cash"

    rets = selected_returns(market, asof, picks, cfg)
    if rets.shape[1] == 0 or len(rets) < cfg.cov_min_bars:
        return pd.Series(dtype=float), picks, "insufficient return history — hold cash"

    weights, note = max_sharpe_weights(rets, risk_free_rate(market, asof, cfg), cfg)
    return weights, picks, note


# ── The back-test ───────────────────────────────────────────────────────────────

def rebalance_dates(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First trading day of each month inside the window."""
    frame = pd.Series(calendar, index=calendar)
    return [g.iloc[0] for _, g in frame.groupby(calendar.to_period("M"))]


def run_strategy(market: MarketData, universe: Sequence[str], cfg: BacktestSettings):
    """Replay the strategy day by day. Returns (equity, trades, plan).

    The month boundary is the only place anything trades. The screen and the
    covariance stop at the last close *before* the first trading day of the month;
    execution is at that day's open. The roll from last month's weights to this
    month's happens in that one trade, so the position carries straight across the
    boundary rather than sitting in cash overnight. In between, equity is marked at
    the close and nothing trades.
    """
    calendar = market.calendar
    rebals = set(rebalance_dates(calendar))

    cash = cfg.principal
    shares: dict[str, float] = {}
    equity_rows, trade_rows, plan_rows = [], [], []

    for day in calendar:
        opens, closes = market.opens.loc[day], market.closes.loc[day]

        if day in rebals:
            asof = market.closes.index[market.closes.index < day][-1]
            weights, picks, note = target_weights(market, asof, universe, cfg)

            mark = {s: float(opens.get(s, np.nan)) for s in shares}
            equity = cash + sum(q * mark[s] for s, q in shares.items()
                                if np.isfinite(mark.get(s, np.nan)))
            investable = equity * (1.0 - cfg.cost_buffer)

            targets = {s: float(w) * investable for s, w in weights.items()
                       if np.isfinite(opens.get(s, np.nan)) and opens.get(s, 0) > 0}

            for sym in sorted(set(shares) | set(targets)):
                ref = float(opens.get(sym, np.nan))
                if not np.isfinite(ref) or ref <= 0:
                    continue
                held = shares.get(sym, 0.0)
                delta = targets.get(sym, 0.0) / ref - held
                if abs(delta * ref) < 1.0:          # skip dust
                    continue
                side = Action.BUY if delta > 0 else Action.SELL
                fill = cfg.costs.fill_price(ref, side)
                notional = delta * fill
                fee = cfg.costs.costs(notional)
                cash -= notional + fee
                shares[sym] = held + delta
                trade_rows.append({"date": day, "symbol": sym, "side": side.value,
                                   "shares": delta, "fill": fill, "notional": notional,
                                   "cost": fee})

            shares = {s: q for s, q in shares.items() if abs(q) > 1e-9}
            plan_rows.append({"rebalance": day, "signal_date": asof, "n_passed": len(picks),
                              "n_held": len(shares), "note": note,
                              "weights": weights.round(6).to_dict()})

        marked = sum(q * float(closes.get(s, np.nan)) for s, q in shares.items()
                     if np.isfinite(closes.get(s, np.nan)))
        equity_rows.append({"date": day, "equity": cash + marked, "cash": cash,
                            "n_positions": len(shares)})

    return (pd.DataFrame(equity_rows).set_index("date"),
            pd.DataFrame(trade_rows), pd.DataFrame(plan_rows))


def buy_and_hold(market: MarketData, symbol: str, cfg: BacktestSettings) -> pd.Series:
    """Principal into one symbol at the first open of the window, marked to close."""
    first = market.calendar[0]
    fill = cfg.costs.fill_price(float(market.opens.loc[first, symbol]), Action.BUY)
    qty = (cfg.principal - cfg.costs.costs(cfg.principal)) / fill
    return (market.closes.loc[market.calendar, symbol] * qty).rename(symbol)


def curves(market: MarketData, equity: pd.DataFrame, cfg: BacktestSettings) -> pd.DataFrame:
    """The strategy's equity beside each benchmark's buy-and-hold, on one calendar.

    Every curve is anchored at the principal on the day before the first trading day,
    so "profit" means profit on the money put in — day one's move, and the cost of
    getting in, are inside the number rather than before it.
    """
    out = pd.DataFrame({"Strategy": equity["equity"]})
    for sym in cfg.benchmarks:
        if sym in market.closes.columns:
            out[sym] = buy_and_hold(market, sym, cfg)
    opening = pd.DataFrame([[cfg.principal] * out.shape[1]], columns=out.columns,
                           index=[market.calendar[0] - pd.Timedelta(days=1)])
    return pd.concat([opening, out])


def equity_metrics(equity: pd.Series) -> dict:
    """Same definitions as ``backtest._compute_metrics``, on an equity curve alone.

    Kept identical so a figure here is comparable to one ``backtest.py`` prints.
    """
    equity = equity.dropna()
    start_val, end_val = float(equity.iloc[0]), float(equity.iloc[-1])
    rets = equity.pct_change().dropna()
    std = float(rets.std())
    years = len(equity) / TRADING_DAYS
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else 0.0
    return {
        "Ending equity": end_val,
        "Profit": end_val - start_val,
        "Total return": end_val / start_val - 1 if start_val > 0 else 0.0,
        "CAGR": cagr,
        "Ann. vol": std * np.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "Sharpe (rf=0)": (float(rets.mean()) / std) * np.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "Max drawdown": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else 0.0,
        "Best day": float(rets.max()),
        "Worst day": float(rets.min()),
    }


def pnl_by_symbol(trades: pd.DataFrame, market: MarketData, equity: pd.DataFrame,
                  cfg: BacktestSettings) -> pd.DataFrame:
    """Realised + open profit per symbol, from the fills alone.

    Cash out minus cash in, plus whatever the position was still worth at the end.
    Summing this column reproduces the strategy's total profit, so an attribution
    chart drawn from it cannot disagree with the equity curve.
    """
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "pnl", "traded_notional", "costs", "end_value"])

    last = market.calendar[-1]
    rows = []
    for sym, grp in trades.groupby("symbol"):
        net_shares = float(grp["shares"].sum())
        cash_flow = float(-(grp["notional"].sum()) - grp["cost"].sum())
        end_value = net_shares * float(market.closes.loc[last, sym]) if abs(net_shares) > 1e-9 else 0.0
        rows.append({
            "symbol": sym,
            "pnl": cash_flow + end_value,
            "traded_notional": float(grp["notional"].abs().sum()),
            "costs": float(grp["cost"].sum()),
            "end_value": end_value,
        })
    return pd.DataFrame(rows).sort_values("pnl", ascending=False).reset_index(drop=True)
