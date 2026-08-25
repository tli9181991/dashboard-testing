import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def parse_bool(val: str, default: bool) -> bool:
    """Helper to safely parse boolean environment variables."""
    if val is None or val == "":
        return default
    return str(val).lower() in ("true", "1", "yes", "y", "t")

# API Keys & Defaults
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
INITIAL_ETH_PRICE = float(os.getenv("INITIAL_ETH_PRICE", "2300.0"))
DEFAULT_POS_SIZE_USD = float(os.getenv("POS_SIZE_USD", "500.0"))
DEFAULT_MAX_POS = int(os.getenv("MAX_POS", "5"))

# Feature Flags (UI Tabs)
SHOW_TAB_BREAKOUT = parse_bool(os.getenv("SHOW_TAB_BREAKOUT"), True)
SHOW_TAB_SCREENER = parse_bool(os.getenv("SHOW_TAB_SCREENER"), True)
SHOW_TAB_SENTIMENT = parse_bool(os.getenv("SHOW_TAB_SENTIMENT"), True)
SHOW_TAB_PORTFOLIO = parse_bool(os.getenv("SHOW_TAB_PORTFOLIO"), False)

# Cache Directories & Local File Paths
CACHE_DIR = Path("./.screen_cache")
CACHE_DIR.mkdir(exist_ok=True)

LOCAL_TOP5_SECTOR_FILE = CACHE_DIR / "top5_stocks_by_sector.csv"
LOCAL_SCREENING_RAW_FILE = CACHE_DIR / "raw_screening_results.csv"

# Screening Parameters
SCREENING_PARAMS = {
    "period": "3y",
    "interval": "1d",
    "market_benchmark": "^GSPC",
    "base_window": 50,
    "prior_window": 50,
    "min_base_depth": 0.12,
    "max_base_depth": 0.33,
    "max_sma50_drift": 0.08,
    "rs_lookback": 126,
    "rs_slope_window": 100,
    "min_avg_volume": 200_000,
    "min_price": 10.0,
    "within_52w_high_pct": 0.25,
    "stage2_slope_window": 20,
    "sector_etf_map": {
        "Communication Services": "XLC",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Financial Services": "XLF",
        "Financial": "XLF",
        "Healthcare": "XLV",
        "Industrials": "XLI",
        "Information Technology": "XLK",
        "Technology": "XLK",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Utilities": "XLU",
    }
}