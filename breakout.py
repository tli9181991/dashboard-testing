import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
from config import CACHE_DIR

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    return tr.rolling(n, min_periods=1).mean()

def _merge_levels(levels: list, price_tol: float) -> list:
    if not levels: return []
    levels = sorted(levels, key=lambda x: (x[1], x[0]))
    merged, cur_times = [], [levels[0][0]]
    cur_price, cur_kind, touches = levels[0][1], levels[0][2], 1

    for t, p, k in levels[1:]:
        if (abs(p - cur_price) <= price_tol) and (k == cur_kind):
            cur_price = (cur_price * touches + p) / (touches + 1)
            touches += 1
            cur_times.append(t)
        else:
            merged.append((cur_times, cur_price, cur_kind, touches))
            cur_times, cur_price, cur_kind, touches = [t], p, k, 1
    merged.append((cur_times, cur_price, cur_kind, touches))
    return merged

def estimate_sr_levels(df: pd.DataFrame, swing_lookback: int = 3, prominence_frac: float = 0.015, atr_window: int = 14) -> list:
    close, high, low = df['Close'], df['High'], df['Low']
    prom = max(1e-6, prominence_frac * float(close.iloc[-1]))
    peaks, _ = find_peaks(high.values, distance=swing_lookback, prominence=prom)
    troughs, _ = find_peaks((-low).values, distance=swing_lookback, prominence=prom)

    levels_raw = [(df.index[i], float(high.iloc[i]), 'resistance') for i in peaks]
    levels_raw += [(df.index[i], float(low.iloc[i]), 'support') for i in troughs]

    avg_atr = float(atr(df, atr_window).mean())
    price_tol = max(1e-6, 1.0 * avg_atr)
    return _merge_levels(levels_raw, price_tol=price_tol)

class SwingBreakoutMonitor:
    def __init__(self, symbol: str, existing_positions: list = None, max_pos: int = 5, pos_size_usd: float = 500.0):
        self.positions = existing_positions if existing_positions is not None else []
        self.max_pos = max_pos
        self.pos_size_usd = pos_size_usd
        self.symbol = symbol
        self.cache_file = CACHE_DIR / f"{self.symbol.replace('-', '_')}_price_history.csv"
        
    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetches market data and caches locally to disk."""
        if not force_refresh and os.path.exists(self.cache_file):
            try:
                df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception:
                pass

        # Fetch fresh data if cache missing or force refreshed
        df = yf.download(self.symbol, period='6mo', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Save to local CSV
        df.to_csv(self.cache_file)
        return df

    def evaluate_market(self, force_refresh: bool = False):
        df = self.fetch_data(force_refresh=force_refresh)
        current_price = float(df['Close'].iloc[-1])
        current_sma20 = float(df['SMA_20'].iloc[-1])
        
        total_pnl = sum(((current_price - p) / p) * self.pos_size_usd for p in self.positions)
        log_msgs = []
        signal = "HOLD"
        
        # Rule 1: Trailing Exit Condition (Close below 20 SMA)
        if current_price < current_sma20:
            log_msgs.append(f"🚨 EXIT SIGNAL: Current Price (${current_price:,.2f}) closed below 20 SMA (${current_sma20:,.2f}).")
            if self.positions:
                total_invested = len(self.positions) * self.pos_size_usd
                log_msgs.append(f"Action: LIQUIDATE ALL positions (Total Capital Invested: ${total_invested:,.2f}).")
                signal = "SELL"
                self.positions = [] 
            return df, current_price, current_sma20, total_pnl, signal, log_msgs

        # Rule 2: Breakout Entry Condition
        log_msgs.append("✅ Trend intact (Price > 20 SMA). Evaluating breakout levels...")
        levels = estimate_sr_levels(df)
        resistances = sorted([l for l in levels if l[2] == 'resistance'], key=lambda x: x[1])
        
        nearest_broken_res = next((price for _, price, _, _ in reversed(resistances) if price < current_price), None)
        next_overhead_res = next((price for _, price, _, _ in resistances if price > current_price), None)

        if next_overhead_res: log_msgs.append(f"🎯 Target Overhead Resistance: ${next_overhead_res:,.2f}")

        if nearest_broken_res:
            pct_above_breakout = (current_price - nearest_broken_res) / nearest_broken_res
            if 0 < pct_above_breakout < 0.02:
                log_msgs.append(f"💡 FRESH BREAKOUT: Price is {pct_above_breakout*100:.2f}% above resistance (${nearest_broken_res:,.2f}).")
                if len(self.positions) < self.max_pos:
                    units = self.pos_size_usd / current_price
                    log_msgs.append(f"🚀 BUY SIGNAL: Triggering entry for ${self.pos_size_usd:,.2f} USD (~{units:.4f} units @ ${current_price:,.2f}).")
                    self.positions.append(current_price)
                    signal = "BUY"
                else:
                    log_msgs.append(f"⚠️ CAP REACHED: Max limit of {self.max_pos} positions active.")
            else:
                log_msgs.append(f"⏳ Holding. Current price maintains buffer above resistance (${nearest_broken_res:,.2f}).")

        return df, current_price, current_sma20, total_pnl, signal, log_msgs