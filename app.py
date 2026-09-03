import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

from config import (
    ACCOUNT_EQUITY, TARGET_VOL, MAX_POSITION_PCT,
    USE_REGIME_GATE, REGIME_BENCHMARK,
    SHOW_TAB_BREAKOUT, SHOW_TAB_SCREENER, SHOW_TAB_SENTIMENT, SHOW_TAB_CHATBOT
)
import data as data_mod
from breakout import SwingBreakoutMonitor
from positions import load_portfolio, conversion_notes
from sizing import SizingParams
from strategy import StrategyParams, add_indicators
from sentiment import get_hourly_sentiment
from screening import get_or_create_sector_stocks
from chat_agent import get_financial_agent

# =============================================================================
# Helper Functions
# =============================================================================
# EMAs for the screener charts come from the same helper the strategy uses, so a
# line on a chart always means what the signal engine means by it.
SCREENER_CHART_PARAMS = StrategyParams(ema_spans=(5, 10, 20))
SCREENER_EMA_STYLE = {
    "EMA_5": ("5 EMA", "#00F0FF", 1),
    "EMA_10": ("10 EMA", "#FF00FF", 1),
    "EMA_20": ("20 EMA", "#00FF00", 1.5),
}


@st.cache_data(ttl=3600)
def fetch_sector_price_history(tickers: tuple, period: str = "6mo") -> dict:
    """Per-ticker OHLCV with EMA overlays, keyed by symbol.

    Prices are left in their own units — each stock gets its own axis, so there is
    nothing to normalise against.
    """
    if not tickers:
        return {}

    raw = yf.download(list(tickers), period=period, interval="1d", progress=False,
                      group_by="ticker", auto_adjust=True)
    if raw is None or raw.empty:
        return {}

    out = {}
    for ticker in tickers:
        try:
            sub = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if sub.empty:
                continue
            out[ticker] = add_indicators(sub, SCREENER_CHART_PARAMS)
        except Exception:
            continue
    return out


def render_price_chart(ticker: str, df: pd.DataFrame, height: int = 360):
    """Candlestick with 5/10/20 EMA overlays, on the stock's own price scale."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, showlegend=False,
    ))
    for column, (label, colour, width) in SCREENER_EMA_STYLE.items():
        if column in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[column], mode="lines", name=label,
                line=dict(color=colour, width=width),
            ))

    last_close = float(df["Close"].iloc[-1])
    fig.update_layout(
        title=f"{ticker} — ${last_close:,.2f}",
        template="plotly_dark", height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        yaxis=dict(title="Price ($)"),
    )
    return fig

# =============================================================================
# Dashboard Initialization
# =============================================================================
st.set_page_config(page_title="Trading Agent & Screener", page_icon="🚀", layout="wide")
st.title("🚀 Automated Portfolio, Screening & Dashboard")

st.sidebar.header("Agent Settings")
refresh_rate = st.sidebar.slider("Refresh Interval (s)", 5, 300, 30)
force_data_refresh = st.sidebar.button("🔄 Force Refresh Market Data")
auto_refresh_paused = st.sidebar.checkbox("⏸️ Pause Auto-Refresh (Turn on when chatting)")

st.sidebar.header("Risk")
account_equity = st.sidebar.number_input("Account Equity ($)", value=float(ACCOUNT_EQUITY), min_value=1000.0, step=1000.0)
target_vol = st.sidebar.slider("Target Volatility per Position", 0.05, 0.50, float(TARGET_VOL), 0.01,
                               help="Annualised volatility budget. Size scales inversely with each name's own volatility.")
max_position_pct = st.sidebar.slider("Max Position Size (% of equity)", 0.05, 1.00, float(MAX_POSITION_PCT), 0.05)
use_regime_gate = st.sidebar.checkbox("Regime gate on new entries", value=USE_REGIME_GATE,
                                      help=f"Block new entries while {REGIME_BENCHMARK} trades below its 200 SMA. Exits are never gated.")

sizing_params = SizingParams(target_vol=target_vol, max_position_pct=max_position_pct)

holdings = load_portfolio()
portfolio_tickers = list(holdings.keys())

if not portfolio_tickers:
    st.warning("⚠️ `portfolio.csv` is empty or missing. Please add tickers to the file to monitor.")
    st.stop()

for note in conversion_notes(holdings):
    st.sidebar.caption(f"ℹ️ {note}")

benchmark_close = None
regime_label = "off"
if use_regime_gate:
    bench = data_mod.load_history(REGIME_BENCHMARK, period="2y")
    if bench.empty:
        st.warning(f"Could not load {REGIME_BENCHMARK}; running without the regime gate.")
    else:
        benchmark_close = bench["Close"]

views = {}
summary_data = []
total_portfolio_pnl = 0.0
failed = []

for ticker, holding in holdings.items():
    monitor = SwingBreakoutMonitor(
        symbol=ticker,
        position=holding.position,
        equity=account_equity,
        sizing_params=sizing_params,
    )
    try:
        view = monitor.evaluate_market(force_refresh=force_data_refresh, benchmark_close=benchmark_close)
    except Exception as exc:
        failed.append(f"{ticker}: {exc}")
        continue

    views[ticker] = view
    total_portfolio_pnl += view.unrealized_pnl
    regime_label = "risk-on" if view.regime_ok else "risk-off"

    summary_data.append({
        "Ticker": ticker,
        "Current Price": f"${view.price:,.2f}",
        "10 SMA": f"${view.decision.sma_exit:,.2f}",
        "Next Resistance": f"${view.decision.next_resistance:,.2f}" if view.decision.next_resistance else "N/A",
        "Avg Cost": f"${holding.position.avg_price:,.2f}" if holding.position.avg_price > 0 else "$0.00",
        "Quantity": f"{holding.quantity:,.6f}".rstrip("0").rstrip(".") if holding.asset_class.value == "crypto" else f"{holding.quantity:,.0f}",
        "Ann. Vol": f"{view.ann_vol:.1%}",
        "Target Size": f"{view.target_quantity:,.4f}".rstrip("0").rstrip(".") if holding.asset_class.value == "crypto" else f"{view.target_quantity:,.0f}",
        "Target $": f"${view.target_notional:,.0f}",
        "Unrealized PnL": f"${view.unrealized_pnl:,.2f}",
        "Long Term": "Yes" if holding.position.long_term else "No",
        "Signal": view.signal,
    })

for message in failed:
    st.error(f"⚠️ {message}")

if not views:
    st.error("No symbols could be evaluated. Check connectivity, then press Force Refresh.")
    st.stop()

portfolio_tickers = list(views.keys())

m1, m2, m3 = st.columns(3)
m1.metric("Monitored Assets", len(portfolio_tickers))
m2.metric("Aggregate Unrealized PnL", f"${total_portfolio_pnl:,.2f}",
          delta_color="normal" if total_portfolio_pnl >= 0 else "inverse")
m3.metric("Market Regime", regime_label.upper() if use_regime_gate and benchmark_close is not None else "GATE OFF")

if use_regime_gate and benchmark_close is not None and regime_label == "risk-off":
    st.info(f"🚦 {REGIME_BENCHMARK} is below its 200 SMA — new entries are vetoed. Exits are unaffected.")

st.markdown("---")

# =============================================================================
# Tabs Navigation
# =============================================================================
tab_titles = []
if SHOW_TAB_BREAKOUT: tab_titles.append("📈 Portfolio Breakout Monitor")
if SHOW_TAB_SCREENER: tab_titles.append("🔍 Stock Selection Screener")
if SHOW_TAB_SENTIMENT: tab_titles.append("🧠 AI Sector & News Sentiment")
if SHOW_TAB_CHATBOT: tab_titles.append("💬 AI Financial Assistant")

rendered_tabs = st.tabs(tab_titles)
tab_index = 0

# TAB 1: Breakout Execution
if SHOW_TAB_BREAKOUT:
    with rendered_tabs[tab_index]:
        st.subheader("Watchlist Summary (portfolio.csv)")
        summary_df = pd.DataFrame(summary_data)
        # FIXED: Changed use_container_width=True to width="stretch"
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_ticker = st.selectbox("Select Asset to Chart:", portfolio_tickers)
            active_df = views[chart_ticker].df
            avg_entry = holdings[chart_ticker].position.avg_price
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=active_df.index, open=active_df['Open'], high=active_df['High'], low=active_df['Low'], close=active_df['Close'], name=chart_ticker))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['SMA_10'], mode='lines', name='10 SMA', line=dict(color='orange', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_5'], mode='lines', name='5 EMA', line=dict(color='#00F0FF', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_10'], mode='lines', name='10 EMA', line=dict(color='#FF00FF', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='#00FF00', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_200'], mode='lines', name='200 EMA', line=dict(color='#FFFFFF', width=2)))
            
            if holdings[chart_ticker].quantity > 0:
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
            st.plotly_chart(fig, width="stretch")
            
        with col2:
            st.subheader(f"Execution Stream: {chart_ticker}")
            active_view = views[chart_ticker]
            if active_view.signal == "BUY":
                st.success("🟢 **BUY SIGNAL**")
                st.caption(f"Vol-targeted size: {active_view.target_quantity:,.4f} units "
                           f"(${active_view.target_notional:,.0f}) at {active_view.ann_vol:.1%} annualised vol.")
            elif active_view.signal == "SELL":
                st.error("🔴 **SELL ALL SIGNAL**")
            else:
                st.info("🟡 **HOLDING**")

            for msg in active_view.logs:
                st.text(f"• {msg}")

            st.caption(f"Bars through {active_view.df.index[-1]:%Y-%m-%d} (completed sessions only).")
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
                # FIXED: Changed use_container_width=True to width="stretch"
                st.dataframe(sector_df[["sector_tag", "ticker", "last_close", "daily_annret", "rs6m_vs_mkt"]], width="stretch", hide_index=True)
                
                tickers = sector_df["ticker"].tolist()
                if tickers:
                    with st.expander(f"View {sector_name} Price Charts (6 Months, 5/10/20 EMA)"):
                        history = fetch_sector_price_history(tuple(tickers))
                        if not history:
                            st.caption("No price data available for this sector right now.")
                        else:
                            chart_cols = st.columns(2)
                            for n, ticker in enumerate(tickers):
                                with chart_cols[n % 2]:
                                    frame = history.get(ticker)
                                    if frame is None or frame.empty:
                                        st.caption(f"No price data for {ticker}.")
                                        continue
                                    st.plotly_chart(
                                        render_price_chart(ticker, frame),
                                        width="stretch",
                                        key=f"screener_{sector_name}_{ticker}",
                                    )
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
        
        # Auth is read from AZURE_INFERENCE_ENDPOINT / AZURE_INFERENCE_CREDENTIAL in config.py
        sentiment_payload = get_hourly_sentiment(sentiment_ticker)
                
        if "error" in sentiment_payload:
            st.warning(sentiment_payload["error"])
        else:
            sdata = sentiment_payload["data"]
            st.markdown(f"### Label: **{sdata['label'].upper()}** (Score: {sdata['score']:.2f})")
            st.progress((sdata['score'] + 1) / 2)
            for art in sentiment_payload["articles"]:
                st.caption(f"📰 **{art['publisher']}**: {art['title']}")
    tab_index += 1

# TAB 4: LangChain AI Financial Assistant
if SHOW_TAB_CHATBOT:
    with rendered_tabs[tab_index]:
        c1, c2 = st.columns([8, 2])
        with c1:
            st.subheader("💬 AI Financial Assistant")
            st.markdown("Ask me to analyze stocks, search the news, or give suggestions based on technicals!")
        with c2:
            if st.button("🗑️ Clear Memory"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        
        # Initialize UI Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Initialize LangChain Memory State
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Render UI Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input logic
        if prompt := st.chat_input("E.g., 'What is the latest news on NVDA and should I buy it?'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent is researching and thinking..."):
                    # FIXED: Called with NO arguments.
                    agent = get_financial_agent() 
                    if not agent:
                        response = "⚠️ Please ensure AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_CREDENTIAL are configured in your .env file."
                    else:
                        try:
                            from langchain_core.messages import HumanMessage, AIMessage
                            
                            # Format memory for LangChain
                            formatted_history = []
                            for msg in st.session_state.chat_history:
                                if msg["role"] == "user":
                                    formatted_history.append(HumanMessage(content=msg["content"]))
                                else:
                                    formatted_history.append(AIMessage(content=msg["content"]))

                            result = agent.invoke({
                                "input": prompt,
                                "chat_history": formatted_history
                            })
                            response = result["output"]
                        except Exception as e:
                            response = f"Agent encountered an error: {str(e)}"
                    
                    st.markdown(response)
            
            # Save to UI memory
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save to LangChain memory
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if not auto_refresh_paused:
    time.sleep(refresh_rate)
    st.rerun()
else:
    st.sidebar.warning("⚠️ Auto-Refresh is paused. Uncheck the box in the sidebar to resume.")