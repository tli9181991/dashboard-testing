import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# Import the new feature flags from config
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
st.title("🚀 Automated Trading, Screening & Dashboard")

# [Keep your existing Sidebar and Top Bar Metrics code here exactly as it was]
# ... (Omitted for brevity, paste your sidebar and metrics logic here) ...

# =============================================================================
# Dynamic Tabs Navigation
# =============================================================================
# Build the list of active tabs based on config flags
tab_titles = []
if SHOW_TAB_BREAKOUT: tab_titles.append("📈 Breakout Monitor")
if SHOW_TAB_SCREENER: tab_titles.append("🔍 Stock Selection Screener")
if SHOW_TAB_SENTIMENT: tab_titles.append("🧠 AI Sector & News Sentiment")
if SHOW_TAB_PORTFOLIO: tab_titles.append("📊 Portfolio Optimization")

if not tab_titles:
    st.warning("All tabs are currently disabled in the configuration.")
    st.stop()

# Generate the tabs dynamically
rendered_tabs = st.tabs(tab_titles)
tab_index = 0

# TAB 1: Breakout Execution
if SHOW_TAB_BREAKOUT:
    with rendered_tabs[tab_index]:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=selected_ticker))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20 SMA', line=dict(color='orange', width=2)))
            for idx, entry in enumerate(st.session_state.portfolio):
                fig.add_hline(y=entry, line_dash="dot", line_color="green", annotation_text=f"Entry ${entry:,.2f}")
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Agent Execution Stream")
            st.caption(f"Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if signal == "BUY": st.success("🟢 **BUY SIGNAL DETECTED**")
            elif signal == "SELL": st.error("🔴 **SELL ALL SIGNAL DETECTED**")
            else: st.info("🟡 **HOLDING**: Trend intact.")
            for msg in logs:
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
                if sector_name == "Unknown":
                    continue
                st.markdown(f"#### 🏛️ Sector: {sector_name}")
                sector_df = cached_top5[cached_top5["sector"] == sector_name]
                st.dataframe(
                    sector_df[["sector_tag", "ticker", "last_close", "daily_annret", "rs6m_vs_mkt"]], 
                    use_container_width=True,
                    hide_index=True
                )
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
        st.subheader(f"AI News Synthesis ({selected_ticker})")
        sentiment_payload = get_hourly_sentiment(selected_ticker, api_key)
        if "error" in sentiment_payload:
            st.warning(sentiment_payload["error"])
        else:
            sdata = sentiment_payload["data"]
            st.markdown(f"### Label: **{sdata['label'].upper()}** (Score: {sdata['score']:.2f})")
            st.progress((sdata['score'] + 1) / 2)
            for art in sentiment_payload["articles"]:
                st.caption(f"📰 **{art['publisher']}**: {art['title']}")
    tab_index += 1

# TAB 4: Portfolio Optimization
if SHOW_TAB_PORTFOLIO:
    with rendered_tabs[tab_index]:
        st.subheader("Markowitz Efficient Frontier & Portfolio Allocation")
        if st.button("Optimize Active Watchlist Portfolio"):
            import yfinance as yf
            from portfolio import compute_portfolio_allocations
            
            with st.spinner("Fetching watchlist data and calculating optimal weights..."):
                data = yf.download(watchlist, period="2y")["Close"].resample("ME").last().pct_change().dropna()
                alloc = compute_portfolio_allocations(data)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("### Max Sharpe Ratio Weights")
                    st.bar_chart(alloc["msr_weights"])
                    st.write(f"Expected Return: {alloc['msr_return']:.2%}")
                with c2:
                    st.write("### Minimum Volatility Weights")
                    st.bar_chart(alloc["vol_weights"])
                    st.write(f"Expected Volatility: {alloc['vol_vol']:.2%}")
    tab_index += 1

time.sleep(refresh_rate)
st.rerun()