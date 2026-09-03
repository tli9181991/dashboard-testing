import json
import time
import pytz
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from screening import load_sp500_symbols, run_screening, get_top5_per_sector

def run_daily_sector_pipeline():
    print(f"\n🚀 Running Daily Pipeline: {datetime.now(pytz.timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M:%S HKT')}")
    symbols = load_sp500_symbols()[:50] # Benchmark scan
    df_passed = run_screening(symbols)
    top5_df = get_top5_per_sector(df_passed)
    
    if not top5_df.empty:
        top5_df.to_csv("top5_stocks_by_sector.csv", index=False)
        print("Saved top 5 stocks by sector.")

def start_hkt_scheduler():
    hkt = pytz.timezone("Asia/Hong_Kong")
    scheduler = BackgroundScheduler(timezone=hkt)
    scheduler.add_job(run_daily_sector_pipeline, trigger="cron", hour=19, minute=0, id="daily_sector_job")
    scheduler.start()
    print("⏰ Scheduler armed: Runs daily at 19:00 HKT.")