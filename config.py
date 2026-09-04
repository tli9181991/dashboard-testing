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
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
AZURE_INFERENCE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT", "")
AZURE_INFERENCE_CREDENTIAL = os.getenv("AZURE_INFERENCE_CREDENTIAL", "")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "DeepSeek-V4-Flash")
INITIAL_ETH_PRICE = float(os.getenv("INITIAL_ETH_PRICE", "2300.0"))
DEFAULT_POS_SIZE_USD = float(os.getenv("POS_SIZE_USD", "500.0"))
DEFAULT_MAX_POS = int(os.getenv("MAX_POS", "5"))

# Feature Flags (UI Tabs)
SHOW_TAB_BREAKOUT = parse_bool(os.getenv("SHOW_TAB_BREAKOUT"), True)
SHOW_TAB_SCREENER = parse_bool(os.getenv("SHOW_TAB_SCREENER"), True)
SHOW_TAB_SENTIMENT = parse_bool(os.getenv("SHOW_TAB_SENTIMENT"), False)
# app.py has always imported SHOW_TAB_CHATBOT; only SHOW_TAB_PORTFOLIO was defined,
# so importing app.py raised ImportError before this was added.
SHOW_TAB_SWING = parse_bool(os.getenv("SHOW_TAB_SWING"), True)
SHOW_TAB_CHATBOT = parse_bool(os.getenv("SHOW_TAB_CHATBOT"), True)
SHOW_TAB_ASSISTANT = parse_bool(os.getenv("SHOW_TAB_ASSISTANT"), True)
#: News older than this is not scored as current sentiment.
NEWS_WINDOW_DAYS = int(os.getenv("NEWS_WINDOW_DAYS", "2"))
SHOW_TAB_PORTFOLIO = parse_bool(os.getenv("SHOW_TAB_PORTFOLIO"), False)

# Risk & sizing defaults (see sizing.py / regime.py)
ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "100000.0"))
TARGET_VOL = float(os.getenv("TARGET_VOL", "0.15"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))
USE_REGIME_GATE = parse_bool(os.getenv("USE_REGIME_GATE"), True)
REGIME_BENCHMARK = os.getenv("REGIME_BENCHMARK", "^GSPC")

# Cache Directories & Local File Paths
CACHE_DIR = Path("./.screen_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Symbols to monitor without holding a position (see watchlist.py).
WATCHLIST_FILE = Path(os.getenv("WATCHLIST_FILE", "./watchlist.json"))

# Assistant conversations are stored here, one JSON file each (see chat_history.py).
CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", "./.chat_history"))

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