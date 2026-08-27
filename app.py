import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

from config import (
    GOOGLE_API_KEY, DEFAULT_POS_SIZE_USD, DEFAULT_MAX_POS,
    SHOW_TAB_BREAKOUT, SHOW_TAB_SCREENER, SHOW_TAB_SENTIMENT, SHOW_TAB_PORTFOLIO
)
from breakout import SwingBreakoutMonitor
from sentiment import get_hourly_sentiment
from screening import get_or_create_sector_stocks

PORTFOLIO_FILE = "portfolio.csv"

# =============================================================================
# Helper Functions
# =============================================================================
def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        df = pd.DataFrame(columns=["ticker", "averaged_price", "total_amount", "long_term"])
        df.to_csv(PORTFOLIO_FILE, index=False)
        return {}
    
    df = pd.read_csv(PORTFOLIO_FILE)
    if "total_ammount" in df.columns:
        df.rename(columns={"total_ammount": "total_amount"}, inplace=True)
        
    port = {}
    for _, row in df.iterrows():
        is_long = str(row.get('long_term', 'False')).strip().lower() in ['true', '1', 'yes', 'y', 't']
        port[row['ticker'].strip().upper()] = {
            "averaged_price": float(row['averaged_price']),
            "total_amount": float(row['total_amount']),
            "long_term": is_long
        }
    return port

@st.cache_data(ttl=3600)
def fetch_normalized_sector_prices(tickers: list) -> pd.DataFrame:
    """Fetches 6M price data for a list of tickers and normalizes to % return."""
    if not tickers: return pd.DataFrame()
    hist = yf.download(tickers, period="6mo", interval="1d", progress=False)
    
    if isinstance(hist.columns, pd.MultiIndex):
        closes = hist["Close"]
    else:
        closes = pd.DataFrame({tickers[0]: hist["Close"]})
    
    closes.dropna(inplace=True)
    if closes.empty: return pd.DataFrame()
    
    # Normalize to Percentage Return from the start of the 6mo period
    return (closes / closes.iloc[0] - 1) * 100

# =============================================================================
# Dashboard Initialization
# =============================================================================
st.set_page_config(page_title="Trading Agent & Screener", page_icon="🚀", layout="wide")
st.title("🚀 Automated Portfolio, Screening & Dashboard")

st.sidebar.header("Agent Settings")
api_key = st.sidebar.text_input("Gemini API Key", value=GOOGLE_API_KEY, type="password")
refresh_rate = st.sidebar.slider("Refresh Interval (s)", 5, 300, 30)
max_pos = st.sidebar.number_input("Max Positions per Asset", value=DEFAULT_MAX_POS, min_value=1)
force_data_refresh = st.sidebar.button("🔄 Force Refresh Market Data")

portfolio_data = load_portfolio()
portfolio_tickers = list(portfolio_data.keys())

if not portfolio_tickers:
    st.warning("⚠️ `portfolio.csv` is empty or missing. Please add tickers to the file to monitor.")
    st.stop()

all_dfs = {}
all_logs = {}
summary_data = []
total_portfolio_pnl = 0.0

for ticker in portfolio_tickers:
    avg_price = portfolio_data[ticker]["averaged_price"]
    tot_amount = portfolio_data[ticker]["total_amount"]
    is_long_term = portfolio_data[ticker]["long_term"]
    
    bot = SwingBreakoutMonitor(
        symbol=ticker, 
        avg_price=avg_price,
        total_amount=tot_amount,
        max_pos=max_pos, 
        pos_size_usd=DEFAULT_POS_SIZE_USD,
        stock_share_size=6,
        long_term=is_long_term
    )
    
    df, current_price, sma10, pnl, signal, next_res, logs = bot.evaluate_market(force_refresh=force_data_refresh)
    
    # We no longer modify portfolio_data or call save_portfolio() here.
    # The agent purely acts as a read-only monitor.

    all_dfs[ticker] = df
    all_logs[ticker] = logs
    total_portfolio_pnl += pnl
    
    summary_data.append({
        "Ticker": ticker,
        "Current Price": f"${current_price:,.2f}",
        "10 SMA": f"${sma10:,.2f}",
        "Next Resistance": f"${next_res:,.2f}" if next_res is not None else "N/A",
        "Avg Cost": f"${portfolio_data[ticker]['averaged_price']:,.2f}" if portfolio_data[ticker]['averaged_price'] > 0 else "$0.00",
        "Total Amount/Shares": f"{portfolio_data[ticker]['total_amount']:.2f}" if bot.is_crypto else f"{int(portfolio_data[ticker]['total_amount'])}",
        "Unrealized PnL": f"${pnl:,.2f}",
        "Long Term": "Yes" if is_long_term else "No",
        "Signal": signal
    })

m1, m2 = st.columns(2)
m1.metric("Total Monitored Assets", len(portfolio_tickers))
m2.metric("Aggregate Portfolio Unrealized PnL", f"${total_portfolio_pnl:,.2f}", delta_color="normal" if total_portfolio_pnl >= 0 else "inverse")

st.markdown("---")

# =============================================================================
# Tabs Navigation
# =============================================================================
tab_titles = []
if SHOW_TAB_BREAKOUT: tab_titles.append("📈 Portfolio Breakout Monitor")
if SHOW_TAB_SCREENER: tab_titles.append("🔍 Stock Selection Screener")
if SHOW_TAB_SENTIMENT: tab_titles.append("🧠 AI Sector & News Sentiment")

rendered_tabs = st.tabs(tab_titles)
tab_index = 0

# TAB 1: Breakout Execution
if SHOW_TAB_BREAKOUT:
    with rendered_tabs[tab_index]:
        st.subheader("Watchlist Summary (portfolio.csv)")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_ticker = st.selectbox("Select Asset to Chart:", portfolio_tickers)
            active_df = all_dfs[chart_ticker]
            avg_entry = portfolio_data[chart_ticker]['averaged_price']
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=active_df.index, open=active_df['Open'], high=active_df['High'], low=active_df['Low'], close=active_df['Close'], name=chart_ticker))
            # fig.add_trace(go.Scatter(x=active_df.index, y=active_df['SMA_10'], mode='lines', name='10 SMA', line=dict(color='orange', width=2, dash='dot')))
            # fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_5'], mode='lines', name='5 EMA', line=dict(color='#00F0FF', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_10'], mode='lines', name='10 EMA', line=dict(color='#FF00FF', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='#00FF00', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_200'], mode='lines', name='200 EMA', line=dict(color='#FFFFFF', width=2)))
            
            if portfolio_data[chart_ticker]['total_amount'] > 0:
                fig.add_hline(y=avg_entry, line_dash="dot", line_color="green", annotation_text=f"Avg Entry ${avg_entry:,.2f}")
                
            fig.update_layout(
                template="plotly_dark", height=600,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=False), type="date"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader(f"Execution Stream: {chart_ticker}")
            active_signal = summary_df.loc[summary_df['Ticker'] == chart_ticker, 'Signal'].values[0]
            if active_signal == "BUY": st.success("🟢 **BUY SIGNAL DETECTED**")
            elif active_signal == "SELL": st.error("🔴 **SELL ALL SIGNAL DETECTED**")
            else: st.info("🟡 **HOLDING**: Trend intact or Long-Term rule active.")
            
            for msg in all_logs[chart_ticker]:
                st.text(f"• {msg}")
    tab_index += 1

# TAB 2: Stock Selection Screener
if SHOW_TAB_SCREENER:
    with rendered_tabs[tab_index]:
        st.subheader("S&P 500 Stage-2 & Relative Strength Screener")
        cached_top5 = get_or_create_sector_stocks(force_rescan=False)
        
        if not cached_top5.empty:
            sectors = cached_top5["sector"].unique()
            for sector_name in sorted(sectors):
                if sector_name == "Unknown": continue
                st.markdown(f"#### 🏛️ Sector: {sector_name}")
                
                sector_df = cached_top5[cached_top5["sector"] == sector_name]
                st.dataframe(sector_df[["sector_tag", "ticker", "last_close", "daily_annret", "rs6m_vs_mkt"]], width="stretch", hide_index=True)
                
                tickers = sector_df["ticker"].tolist()
                if tickers:
                    with st.expander(f"View {sector_name} Relative Performance Chart (6 Months)"):
                        sec_prices = fetch_normalized_sector_prices(tickers)
                        if not sec_prices.empty:
                            fig_sec = go.Figure()
                            for t in tickers:
                                if t in sec_prices.columns:
                                    fig_sec.add_trace(go.Scatter(x=sec_prices.index, y=sec_prices[t], mode='lines', name=t))
                            fig_sec.update_layout(
                                title=f"{sector_name} Top 5 - 6 Month Relative Return (%)",
                                template="plotly_dark", height=400,
                                xaxis_title="Date", yaxis_title="Performance (%)"
                            )
                            st.plotly_chart(fig_sec, use_container_width=True)
                st.markdown("---")
            
        if st.button("Run Fresh Full Stock Rescan (Warning: Takes ~3-5 mins)"):
            with st.spinner("Scraping S&P 500 and executing fresh screening..."):
                get_or_create_sector_stocks(force_rescan=True)
                st.rerun()
    tab_index += 1

# TAB 3: AI Sentiment 
if SHOW_TAB_SENTIMENT:
    with rendered_tabs[tab_index]:
        sentiment_ticker = st.selectbox("Select Portfolio Asset for AI Analysis:", portfolio_tickers, key="sentiment_box")
        st.subheader(f"AI News Synthesis ({sentiment_ticker})")
        sentiment_payload = get_hourly_sentiment(sentiment_ticker, api_key)
        if "error" in sentiment_payload:
            st.warning(sentiment_payload["error"])
        else:
            sdata = sentiment_payload["data"]
            st.markdown(f"### Label: **{sdata['label'].upper()}** (Score: {sdata['score']:.2f})")
            st.progress((sdata['score'] + 1) / 2)
            for art in sentiment_payload["articles"]:
                st.caption(f"📰 **{art['publisher']}**: {art['title']}")
    tab_index += 1

time.sleep(refresh_rate)
st.rerun()