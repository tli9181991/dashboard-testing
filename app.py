import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from config import (
    GOOGLE_API_KEY, INITIAL_ETH_PRICE, DEFAULT_POS_SIZE_USD, DEFAULT_MAX_POS,
    SHOW_TAB_BREAKOUT, SHOW_TAB_SCREENER, SHOW_TAB_SENTIMENT, SHOW_TAB_PORTFOLIO
)
from breakout import SwingBreakoutMonitor
from sentiment import get_hourly_sentiment
from screening import get_or_create_sector_stocks

# =============================================================================
# Dashboard Initialization
# =============================================================================
st.set_page_config(page_title="Trading Agent & Screener", page_icon="🚀", layout="wide")
st.title("🚀 Automated Portfolio, Screening & Dashboard")

# Sidebar Configuration
st.sidebar.header("Agent Settings")
api_key = st.sidebar.text_input("Gemini API Key", value=GOOGLE_API_KEY, type="password")

st.sidebar.header("Watchlist Controls")
watchlist_input = st.sidebar.text_area("Watchlist (one per line)", "ETH-USD\nBTC-USD\nNVDA\nAAPL\nTSLA\nDELL\nPLTR")
watchlist = [t.strip().upper() for t in watchlist_input.split("\n") if t.strip()]

refresh_rate = st.sidebar.slider("Refresh Interval (s)", 5, 300, 30)
max_pos = st.sidebar.number_input("Max Positions per Asset", value=DEFAULT_MAX_POS, min_value=1)
force_data_refresh = st.sidebar.button("🔄 Force Refresh Market Data")

# Multi-Asset Session State Initialization
if "portfolios" not in st.session_state:
    st.session_state.portfolios = {}
    
for ticker in watchlist:
    if ticker not in st.session_state.portfolios:
        st.session_state.portfolios[ticker] = [INITIAL_ETH_PRICE] if ticker == "ETH-USD" else []

# Process All Tickers for Summary
all_dfs = {}
all_logs = {}
summary_data = []
total_portfolio_pnl = 0.0

for ticker in watchlist:
    bot = SwingBreakoutMonitor(
        symbol=ticker, 
        existing_positions=st.session_state.portfolios[ticker], 
        max_pos=max_pos, 
        pos_size_usd=DEFAULT_POS_SIZE_USD,
        stock_share_size=6
    )
    df, current_price, sma10, pnl, signal, logs = bot.evaluate_market(force_refresh=force_data_refresh)
    
    st.session_state.portfolios[ticker] = bot.positions
    all_dfs[ticker] = df
    all_logs[ticker] = logs
    total_portfolio_pnl += pnl
    
    summary_data.append({
        "Ticker": ticker,
        "Price": f"${current_price:,.2f}",
        "10 SMA": f"${sma10:,.2f}",
        "Active Pos": len(bot.positions),
        "Unrealized PnL": f"${pnl:,.2f}",
        "Signal": signal
    })

# Top Bar Metrics
m1, m2 = st.columns(2)
m1.metric("Total Monitored Assets", len(watchlist))
m2.metric("Aggregate Portfolio Unrealized PnL", f"${total_portfolio_pnl:,.2f}", delta_color="normal" if total_portfolio_pnl >= 0 else "inverse")

st.markdown("---")

# =============================================================================
# Dynamic Tabs Navigation
# =============================================================================
tab_titles = []
if SHOW_TAB_BREAKOUT: tab_titles.append("📈 Portfolio Breakout Monitor")
if SHOW_TAB_SCREENER: tab_titles.append("🔍 Stock Selection Screener")
if SHOW_TAB_SENTIMENT: tab_titles.append("🧠 AI Sector & News Sentiment")
if SHOW_TAB_PORTFOLIO: tab_titles.append("📊 Portfolio Optimization")

if not tab_titles:
    st.warning("All tabs are currently disabled in the configuration.")
    st.stop()

rendered_tabs = st.tabs(tab_titles)
tab_index = 0

# TAB 1: Breakout Execution
if SHOW_TAB_BREAKOUT:
    with rendered_tabs[tab_index]:
        st.subheader("Watchlist Summary")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_ticker = st.selectbox("Select Asset to Chart:", watchlist)
            active_df = all_dfs[chart_ticker]
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=active_df.index, open=active_df['Open'], high=active_df['High'], low=active_df['Low'], close=active_df['Close'], name=chart_ticker))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['SMA_10'], mode='lines', name='10 SMA', line=dict(color='orange', width=2)))
            
            for idx, entry in enumerate(st.session_state.portfolios[chart_ticker]):
                fig.add_hline(y=entry, line_dash="dot", line_color="green", annotation_text=f"Entry ${entry:,.2f}")
                
            fig.update_layout(
                template="plotly_dark", 
                height=550,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=False),
                    type="date"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader(f"Execution Stream: {chart_ticker}")
            st.caption(f"Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            active_signal = summary_df.loc[summary_df['Ticker'] == chart_ticker, 'Signal'].values[0]
            if active_signal == "BUY": st.success("🟢 **BUY SIGNAL DETECTED**")
            elif active_signal == "SELL": st.error("🔴 **SELL ALL SIGNAL DETECTED**")
            else: st.info("🟡 **HOLDING**: Trend intact.")
            
            for msg in all_logs[chart_ticker]:
                st.text(f"• {msg}")
    tab_index += 1

# TAB 2: Stock Selection Screener
if SHOW_TAB_SCREENER:
    with rendered_tabs[tab_index]:
        st.subheader("S&P 500 Stage-2 & Relative Strength Screener")
        cached_top5 = get_or_create_sector_stocks(force_rescan=False)
        if not cached_top5.empty:
            st.success("📁 Displaying sector top stocks (Loaded from local disk).")
            sectors = cached_top5["sector"].unique()
            for sector_name in sorted(sectors):
                if sector_name == "Unknown": continue
                st.markdown(f"#### 🏛️ Sector: {sector_name}")
                st.dataframe(cached_top5[cached_top5["sector"] == sector_name][["sector_tag", "ticker", "last_close", "daily_annret", "rs6m_vs_mkt"]], use_container_width=True, hide_index=True)
                st.markdown("---")
        else:
            st.warning("No stocks passed the strict filter during launch scan. Try forcing a fresh rescan.")
            
        if st.button("Run Fresh Full Stock Rescan (Warning: Takes ~3-5 mins)"):
            with st.spinner("Scraping S&P 500 and executing fresh screening..."):
                fresh_top5 = get_or_create_sector_stocks(force_rescan=True)
                st.success("Fresh screening complete and saved to local disk!")
                st.rerun()
    tab_index += 1

# TAB 3: AI Sentiment
if SHOW_TAB_SENTIMENT:
    with rendered_tabs[tab_index]:
        sentiment_ticker = st.selectbox("Select Asset for AI Analysis:", watchlist, key="sentiment_box")
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

# TAB 4: Portfolio Optimization (Commented logic safely bypassed if disabled)
if SHOW_TAB_PORTFOLIO:
    with rendered_tabs[tab_index]:
        st.subheader("Markowitz Efficient Frontier & Portfolio Allocation")
        # Optimization Code Execution here...
    tab_index += 1

time.sleep(refresh_rate)
st.rerun()