import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup as BS
from dataclasses import dataclass
from typing import Dict, List, Optional
from config import SCREENING_PARAMS, LOCAL_TOP5_SECTOR_FILE, LOCAL_SCREENING_RAW_FILE, CACHE_DIR

@dataclass
class ScreenResult:
    ticker: str
    last_close: float
    sma200: float
    within_52w_high_pct: float
    rs6m_vs_mkt: float
    sector: str
    sector_etf: str
    sector_outperforms: bool
    avg_vol50: float
    daily_annret: float
    ann_vol: float
    passed: bool

def load_sp500_symbols() -> List[str]:
    sp500_symbols = []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
        soup = BS(res.text, "html.parser")
        table = soup.find('table', {'id': 'constituents'})
        for row in table.find_all('tr')[1:]:
            sym = row.find_all('td')[0].text.strip().replace('.', '-')
            sp500_symbols.append(sym)
    except Exception as e:
        print(f"Error scraping S&P 500: {e}")
    # Return full universe instead of slicing [:30]
    return sp500_symbols if sp500_symbols else ["NVDA", "AAPL", "MSFT", "JPM", "XOM"]

def yf_info_cached(ticker: str) -> dict:
    """Caches Ticker.info to avoid repeated network calls and speed up sector matching."""
    f = CACHE_DIR / f"info_{ticker}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    try:
        info = yf.Ticker(ticker).info
        f.write_text(json.dumps(info))
        return info
    except Exception:
        return {}

def download_history(tickers: List[str], params=SCREENING_PARAMS) -> Dict[str, pd.DataFrame]:
    if not tickers: return {}
    df = yf.download(tickers=tickers, period=params["period"], interval=params["interval"], group_by="ticker", threads=True, progress=False)
    data = {}
    for t in tickers:
        try:
            sub = df[t].dropna(how="all")[["Open", "High", "Low", "Close", "Volume"]].dropna()
            data[t] = sub
        except Exception:
            continue
    return data

def evaluate_ticker(t: str, px: pd.DataFrame, mpx: pd.Series, params=SCREENING_PARAMS) -> Optional[ScreenResult]:
    if px.empty or px["Close"].dropna().empty: return None
    close, vol = px["Close"].dropna(), px["Volume"].fillna(0)
    last_close = float(close.iloc[-1])
    
    if last_close < params["min_price"]: return None
    avg_vol50 = float(vol.tail(50).mean())
    if avg_vol50 < params["min_avg_volume"]: return None

    ret = close.pct_change().dropna()
    ann_vol = float(ret.std() * np.sqrt(252))
    cum_ret = float((1 + ret).prod() - 1)
    daily_annret = float(((1 + cum_ret)**(252 / max(len(ret), 1))) - 1)

    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    
    stage2_ok = bool(
        last_close > sma150.iloc[-1] and 
        last_close > sma200.iloc[-1] and 
        (sma150.iloc[-1] > sma150.iloc[-20]) and 
        (sma200.iloc[-1] > sma200.iloc[-20])
    )

    high_52w = float(close.tail(252).max())
    within_high_pct = 1.0 - (last_close / high_52w)
    within_high_ok = within_high_pct <= params["within_52w_high_pct"]
    above_200_ok = last_close > float(sma200.iloc[-1])

    mkt = mpx.reindex(close.index).dropna()
    if len(mkt) < params["rs_lookback"]: return None
    
    stock_ret = float(close.iloc[-1] / close.iloc[-params["rs_lookback"]] - 1)
    mkt_ret = float(mkt.iloc[-1] / mkt.iloc[-params["rs_lookback"]] - 1)
    rs6m_vs_mkt = stock_ret - mkt_ret
    rs_ok = rs6m_vs_mkt > 0

    passed = all([stage2_ok, rs_ok, above_200_ok, within_high_ok])

    # Dynamic Sector Resolution (Fixes the XLK bug)
    info = yf_info_cached(t)
    sector_name = info.get("sector") or info.get("industry") or "Unknown"
    sector_etf = ""
    for key, val in params["sector_etf_map"].items():
        if key.lower() in sector_name.lower():
            sector_etf = val
            break

    return ScreenResult(
        ticker=t, last_close=last_close, sma200=float(sma200.iloc[-1]),
        within_52w_high_pct=within_high_pct, rs6m_vs_mkt=rs6m_vs_mkt,
        sector=sector_name, sector_etf=sector_etf, sector_outperforms=True,
        avg_vol50=avg_vol50, daily_annret=daily_annret, ann_vol=ann_vol, passed=passed
    )

def run_screening(tickers: List[str]) -> pd.DataFrame:
    mkt_hist = yf.download(SCREENING_PARAMS["market_benchmark"], period=SCREENING_PARAMS["period"], progress=False)
    if isinstance(mkt_hist.columns, pd.MultiIndex):
        mkt_hist.columns = mkt_hist.columns.get_level_values(0)
    market_close = mkt_hist["Close"].dropna()
    
    hist_map = download_history(tickers)
    results = []
    for t, dfpx in hist_map.items():
        res = evaluate_ticker(t, dfpx, market_close)
        if res: results.append(res.__dict__)
    
    df = pd.DataFrame(results)
    if not df.empty:
        df_passed = df[df["passed"]].sort_values("daily_annret", ascending=False)
        df_passed.to_csv(LOCAL_SCREENING_RAW_FILE, index=False)
        return df_passed
    return df

def get_top5_per_sector(df_passed: pd.DataFrame) -> pd.DataFrame:
    if df_passed.empty: return pd.DataFrame()
    sorted_df = df_passed.sort_values(["sector", "daily_annret"], ascending=[True, False])
    top5 = sorted_df.groupby("sector").head(5).copy()
    top5["sector_rank"] = top5.groupby("sector").cumcount() + 1
    top5["sector_tag"] = top5.apply(lambda r: f"{r['sector_etf']}_Top{r['sector_rank']}" if r['sector_etf'] else f"Top{r['sector_rank']}", axis=1)
    
    top5.to_csv(LOCAL_TOP5_SECTOR_FILE, index=False)
    return top5

def get_or_create_sector_stocks(force_rescan: bool = False) -> pd.DataFrame:
    if not force_rescan and os.path.exists(LOCAL_TOP5_SECTOR_FILE):
        try:
            df = pd.read_csv(LOCAL_TOP5_SECTOR_FILE)
            if not df.empty:
                return df
        except Exception:
            pass

    # Note: Scanning all 500 symbols takes ~3-5 minutes depending on your network.
    symbols = load_sp500_symbols()
    df_passed = run_screening(symbols)
    top5 = get_top5_per_sector(df_passed)
    return top5