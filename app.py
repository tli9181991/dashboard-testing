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
    SHOW_TAB_BREAKOUT, SHOW_TAB_SCREENER, SHOW_TAB_SWING, SHOW_TAB_SENTIMENT,
    SHOW_TAB_CHATBOT, CHAT_HISTORY_DIR, WATCHLIST_FILE
)
import data as data_mod
from breakout import SwingBreakoutMonitor
from positions import load_portfolio, conversion_notes
from sizing import SizingParams
from strategy import StrategyParams, add_indicators
from chat_history import ChatHistoryStore, make_message
from watchlist import WatchlistStore
import swing_screener as swing
from strategy import AssetClass, Position
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


@st.cache_data(ttl=3600, show_spinner=False)
def run_swing_scan(tickers: tuple, overrides: tuple, use_demo: bool, demo_size: int):
    """Run the Swing Universe Funnel. Cached — a full scan is expensive.

    ``overrides`` is a tuple of (key, value) pairs so the cache key stays hashable.
    """
    cfg = dict(swing.CFG)
    cfg.update(dict(overrides))

    if use_demo:
        bars = swing.load_demo(n_names=demo_size, seed=11)
        sector_data = None
    else:
        bars = swing.load_yfinance(sorted(set(list(tickers) + ["SPY"])))
        sector_data = swing.load_yfinance(swing.SECTOR_ETFS)

    if "SPY" not in bars:
        return None, {"stage": "no SPY series — the regime layer needs one"}, cfg

    spy = bars.pop("SPY")
    if not bars:
        return None, {"stage": "no symbols survived loading"}, cfg

    out, ctx = swing.run_scan(bars, spy, cfg, sector_data=sector_data)
    return out, ctx, cfg


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
watchlist_store = WatchlistStore(WATCHLIST_FILE)

# Watchlist names are monitored with a flat position, so the strategy reports the
# entry signal rather than an exit. A symbol held in portfolio.csv stays a holding.
all_watched = watchlist_store.symbols()
watched_set = set(all_watched)
watchlist_symbols = [s for s in all_watched if s not in holdings]

monitored: dict[str, dict] = {
    symbol: {"position": h.position, "asset_class": h.asset_class, "held": True}
    for symbol, h in holdings.items()
}
for symbol in watchlist_symbols:
    monitored[symbol] = {
        "position": Position(),
        "asset_class": AssetClass.infer(symbol),
        "held": False,
    }

if not monitored:
    st.warning(
        "⚠️ Nothing to monitor yet. Add positions to `portfolio.csv`, "
        "or pick stocks from the Screener tab to build a watchlist."
    )
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

def _format_units(quantity: float, asset_class) -> str:
    if asset_class is AssetClass.CRYPTO:
        return f"{quantity:,.6f}".rstrip("0").rstrip(".")
    return f"{quantity:,.0f}"


views = {}
holding_rows = []
watchlist_rows = []
total_portfolio_pnl = 0.0
failed = []

for ticker, entry in monitored.items():
    position, asset_class, held = entry["position"], entry["asset_class"], entry["held"]
    monitor = SwingBreakoutMonitor(
        symbol=ticker,
        position=position,
        equity=account_equity,
        sizing_params=sizing_params,
    )
    try:
        view = monitor.evaluate_market(force_refresh=force_data_refresh, benchmark_close=benchmark_close)
    except Exception as exc:
        failed.append(f"{ticker}: {exc}")
        continue

    views[ticker] = view
    regime_label = "risk-on" if view.regime_ok else "risk-off"

    common = {
        "Ticker": ticker,
        "Current Price": f"${view.price:,.2f}",
        "10 SMA": f"${view.decision.sma_exit:,.2f}",
        "Next Resistance": f"${view.decision.next_resistance:,.2f}" if view.decision.next_resistance else "N/A",
        "Ann. Vol": f"{view.ann_vol:.1%}",
        "Target Size": _format_units(view.target_quantity, asset_class),
        "Target $": f"${view.target_notional:,.0f}",
        "Signal": view.signal,
    }

    if held:
        total_portfolio_pnl += view.unrealized_pnl
        holding_rows.append({
            **common,
            "Avg Cost": f"${position.avg_price:,.2f}" if position.avg_price > 0 else "$0.00",
            "Quantity": _format_units(position.quantity, asset_class),
            "Unrealized PnL": f"${view.unrealized_pnl:,.2f}",
            "Long Term": "Yes" if position.long_term else "No",
        })
    else:
        watchlist_rows.append(common)

for message in failed:
    st.error(f"⚠️ {message}")

if not views:
    st.error("No symbols could be evaluated. Check connectivity, then press Force Refresh.")
    st.stop()

monitored_tickers = list(views.keys())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Holdings", len(holding_rows))
m2.metric("Watchlist", len(watchlist_rows))
m3.metric("Aggregate Unrealized PnL", f"${total_portfolio_pnl:,.2f}",
          delta_color="normal" if total_portfolio_pnl >= 0 else "inverse")
m4.metric("Market Regime", regime_label.upper() if use_regime_gate and benchmark_close is not None else "GATE OFF")

if use_regime_gate and benchmark_close is not None and regime_label == "risk-off":
    st.info(f"🚦 {REGIME_BENCHMARK} is below its 200 SMA — new entries are vetoed. Exits are unaffected.")

st.markdown("---")

# =============================================================================
# Tabs Navigation
# =============================================================================
tab_titles = []
if SHOW_TAB_BREAKOUT: tab_titles.append("📈 Monitoring")
if SHOW_TAB_SCREENER: tab_titles.append("🔍 Stock Selection Screener")
if SHOW_TAB_SWING: tab_titles.append("🎯 Swing Screener")
if SHOW_TAB_SENTIMENT: tab_titles.append("🧠 AI Sector & News Sentiment")
if SHOW_TAB_CHATBOT: tab_titles.append("💬 AI Financial Assistant")

rendered_tabs = st.tabs(tab_titles)
tab_index = 0

# TAB 1: Monitoring (holdings + watchlist)
if SHOW_TAB_BREAKOUT:
    with rendered_tabs[tab_index]:
        st.subheader("Holdings (portfolio.csv)")
        if holding_rows:
            st.dataframe(pd.DataFrame(holding_rows), width="stretch", hide_index=True)
        else:
            st.caption("No positions in `portfolio.csv`.")

        st.subheader("Watchlist")
        if watchlist_rows:
            st.dataframe(pd.DataFrame(watchlist_rows), width="stretch", hide_index=True)
            st.caption(
                "Watchlist names are evaluated with no position, so BUY marks a fresh "
                "entry signal. They carry no PnL until you add them to `portfolio.csv`."
            )
            st.caption("Remove from watchlist:")
            remove_cols = st.columns(min(len(watchlist_rows), 6))
            for n, row in enumerate(watchlist_rows):
                watched_ticker = row["Ticker"]
                with remove_cols[n % len(remove_cols)]:
                    if st.button(f"✕ {watched_ticker}", key=f"wl_remove_{watched_ticker}",
                                 width="stretch",
                                 help=f"Stop watching {watched_ticker}"):
                        watchlist_store.remove(watched_ticker)
                        st.toast(f"Removed {watched_ticker} from the watchlist.")
                        st.rerun()
        else:
            st.caption("Empty. Pick stocks from the 🔍 Screener tab to start watching them.")

        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_ticker = st.selectbox("Select Asset to Chart:", monitored_tickers)
            active_df = views[chart_ticker].df
            avg_entry = holdings[chart_ticker].position.avg_price if chart_ticker in holdings else 0.0
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=active_df.index, open=active_df['Open'], high=active_df['High'], low=active_df['Low'], close=active_df['Close'], name=chart_ticker))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['SMA_10'], mode='lines', name='10 SMA', line=dict(color='orange', width=2, dash='dot')))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_5'], mode='lines', name='5 EMA', line=dict(color='#00F0FF', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_10'], mode='lines', name='10 EMA', line=dict(color='#FF00FF', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_20'], mode='lines', name='20 EMA', line=dict(color='#00FF00', width=1)))
            fig.add_trace(go.Scatter(x=active_df.index, y=active_df['EMA_200'], mode='lines', name='200 EMA', line=dict(color='#FFFFFF', width=2)))
            
            if chart_ticker in holdings and holdings[chart_ticker].quantity > 0:
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

                                    # Add / remove this name from the monitoring tab.
                                    if ticker in holdings:
                                        st.button(
                                            f"✅ {ticker} — already a holding",
                                            key=f"watch_{sector_name}_{ticker}",
                                            disabled=True, width="stretch",
                                        )
                                    elif ticker in watched_set:
                                        if st.button(
                                            f"👁️ {ticker} — watching (click to remove)",
                                            key=f"watch_{sector_name}_{ticker}",
                                            width="stretch",
                                        ):
                                            watchlist_store.remove(ticker)
                                            st.toast(f"Removed {ticker} from the watchlist.")
                                            st.rerun()
                                    else:
                                        if st.button(
                                            f"➕ Add {ticker} to watchlist",
                                            key=f"watch_{sector_name}_{ticker}",
                                            type="primary", width="stretch",
                                        ):
                                            watchlist_store.add(ticker, sector=sector_name, source="screener")
                                            st.toast(f"Added {ticker} to the watchlist.")
                                            st.rerun()

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

# TAB 3: Swing Universe Funnel
if SHOW_TAB_SWING:
    with rendered_tabs[tab_index]:
        st.subheader("🎯 Swing Universe Funnel")
        st.markdown(
            "Liquidity gate → tradability → regime → setup (momentum pullback or "
            "ICT 2022 on daily bars) → earnings gate → sizing → ranking."
        )

        sc1, sc2 = st.columns([3, 2])
        with sc1:
            universe_choice = st.radio(
                "Universe", ["Demo (synthetic)", "Portfolio + Watchlist", "Custom list"],
                horizontal=True, key="swing_universe",
                help="Demo runs offline on generated bars. The others fetch ~3y of daily bars.",
            )
        with sc2:
            demo_size = st.slider("Demo universe size", 40, 300, 200, 20,
                                  disabled=universe_choice != "Demo (synthetic)")

        custom_text = ""
        if universe_choice == "Custom list":
            custom_text = st.text_input(
                "Tickers (comma separated)", value="AAPL,MSFT,NVDA,AMD,AVGO,TSM,META,GOOGL",
                key="swing_custom",
            )

        with st.expander("Thresholds"):
            t1, t2, t3 = st.columns(3)
            with t1:
                swing_equity = st.number_input("Account equity ($)", value=float(account_equity),
                                               min_value=1000.0, step=1000.0, key="swing_equity")
                risk_per_trade = st.slider("Risk per trade", 0.0025, 0.02,
                                           float(swing.CFG["risk_per_trade"]), 0.0025,
                                           format="%.4f")
            with t2:
                keep_pct = st.slider("Tradability keep %", 0.10, 1.00,
                                     float(swing.CFG["tradability_keep_pct"]), 0.05,
                                     help="Fraction of gate survivors kept before the setup layer.")
                max_positions = st.number_input("Max positions", 1, 20,
                                                int(swing.CFG["max_positions"]))
            with t3:
                adr_lo, adr_hi = st.slider("ADR% band", 0.5, 12.0,
                                           (float(swing.CFG["adr_min"]), float(swing.CFG["adr_max"])), 0.1)
                er_min = st.slider("Min efficiency ratio", 0.0, 0.8,
                                   float(swing.CFG["er_min"]), 0.05)
            st.caption(
                "Every constant here is a starting value the author flags as unvalidated. "
                "Sweep them in a backtest before trading them."
            )

        overrides = (
            ("account_equity", float(swing_equity)),
            ("risk_per_trade", float(risk_per_trade)),
            ("tradability_keep_pct", float(keep_pct)),
            ("max_positions", int(max_positions)),
            ("adr_min", float(adr_lo)),
            ("adr_max", float(adr_hi)),
            ("er_min", float(er_min)),
        )

        if universe_choice == "Portfolio + Watchlist":
            swing_tickers = tuple(sorted(set(list(holdings) + all_watched)))
        elif universe_choice == "Custom list":
            swing_tickers = tuple(sorted({t.strip().upper() for t in custom_text.split(",") if t.strip()}))
        else:
            swing_tickers = ()

        use_demo = universe_choice == "Demo (synthetic)"
        if not use_demo and not swing_tickers:
            st.info("Add some symbols to scan, or switch to the demo universe.")
        else:
            if st.button("🔄 Rescan", key="swing_rescan"):
                run_swing_scan.clear()

            with st.spinner("Running the funnel..."):
                try:
                    swing_out, swing_ctx, swing_cfg = run_swing_scan(
                        swing_tickers, overrides, use_demo, int(demo_size)
                    )
                except Exception as exc:
                    swing_out, swing_ctx, swing_cfg = None, {"stage": f"scan failed: {exc}"}, {}

            if swing_out is None:
                st.error(f"⚠️ {swing_ctx.get('stage', 'scan produced nothing')}")
            else:
                regime = str(swing_ctx.get("regime", "unknown"))
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("As of", str(swing_ctx.get("asof", "—")))
                r2.metric("Regime", regime.replace("_", " ").upper())
                breadth = swing_ctx.get("breadth")
                r3.metric("Breadth", f"{breadth:.0%}" if breadth == breadth else "—")
                r4.metric("Slots", swing_ctx.get("slots", 0))

                st.markdown(
                    f"**Funnel** — {swing_ctx.get('universe', 0)} universe → "
                    f"{swing_ctx.get('gated', 0)} liquidity gate → "
                    f"{swing_ctx.get('floor_ok', 0)} tradability floor → "
                    f"{swing_ctx.get('tradable', 0)} tradable → "
                    f"**{len(swing_out)} with a setup**"
                )

                if regime == "risk_off":
                    st.warning("🚫 Regime is risk-off. The long side is closed and no candidates are produced.")

                if swing_out.empty:
                    st.info(f"No candidates — {swing_ctx.get('stage', 'nothing qualified')}.")
                else:
                    display_cols = ["ticker", "variant", "close", "entry", "stop", "tp1",
                                    "tp2", "r_tp1", "shares", "risk_$", "adr%", "er", "gap", "score"]
                    shown = swing_out[[c for c in display_cols if c in swing_out.columns]].copy()
                    shown["score"] = shown["score"].round(2)
                    shown["close"] = shown["close"].round(2)
                    st.dataframe(shown, width="stretch", hide_index=True)

                    slots = int(swing_ctx.get("slots", 0)) or len(swing_out)
                    st.markdown(f"#### Book — top {min(slots, len(swing_out))} by composite score")
                    for _, cand in swing_out.head(slots).iterrows():
                        ticker = str(cand["ticker"])
                        risk_per_share = float(cand["entry"]) - float(cand["stop"])
                        with st.container(border=True):
                            h1, h2 = st.columns([5, 2])
                            with h1:
                                st.markdown(f"**{ticker}** · {cand['variant']} — {cand['note']}")
                                st.caption(
                                    f"Entry ${cand['entry']:,.2f} · Stop ${cand['stop']:,.2f} "
                                    f"(${risk_per_share:,.2f}/share) · TP1 ${cand['tp1']:,.2f} "
                                    f"· TP2 ${cand['tp2']:,.2f} · {cand['shares']:,} shares "
                                    f"risking ${cand['risk_$']:,.0f}"
                                )
                            with h2:
                                if ticker in holdings:
                                    st.button("✅ already a holding", key=f"swing_add_{ticker}",
                                              disabled=True, width="stretch")
                                elif ticker in watched_set:
                                    st.button("👁️ on watchlist", key=f"swing_add_{ticker}",
                                              disabled=True, width="stretch")
                                else:
                                    if st.button(f"➕ Watch {ticker}", key=f"swing_add_{ticker}",
                                                 type="primary", width="stretch"):
                                        watchlist_store.add(ticker, sector=str(cand["variant"]),
                                                            source="swing_screener")
                                        st.toast(f"Added {ticker} to the watchlist.")
                                        st.rerun()

                    st.download_button(
                        "⬇️ Download candidates as CSV",
                        swing_out.to_csv(index=False).encode("utf-8"),
                        file_name=f"swing_candidates_{swing_ctx.get('asof', 'scan')}.csv",
                        mime="text/csv",
                    )

                rejects = swing_ctx.get("rejects") or {}
                if rejects:
                    with st.expander(f"Rejected by the liquidity gate ({len(rejects)})"):
                        st.dataframe(
                            pd.DataFrame(sorted(rejects.items()), columns=["Ticker", "Reason"]),
                            width="stretch", hide_index=True,
                        )
    tab_index += 1

# TAB 4: AI Sentiment 
if SHOW_TAB_SENTIMENT:
    with rendered_tabs[tab_index]:
        sentiment_ticker = st.selectbox("Select Asset for AI Analysis:", monitored_tickers, key="sentiment_box")
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

# TAB 5: LangChain AI Financial Assistant
if SHOW_TAB_CHATBOT:
    with rendered_tabs[tab_index]:
        chat_store = ChatHistoryStore(CHAT_HISTORY_DIR)

        def _open_conversation(conversation):
            """Point session state — and the picker widget — at one conversation."""
            st.session_state.conversation_id = conversation.id
            st.session_state.messages = conversation.messages
            # A keyed widget's stored value wins over its `index` argument, so the
            # picker has to be moved explicitly or it snaps back to the old thread.
            st.session_state.conversation_picker = conversation.id

        # Reopen whatever conversation was last in use. session_state is in-memory
        # only, so without this every restart started from a blank thread.
        if "conversation_id" not in st.session_state:
            _open_conversation(chat_store.load_current_or_create())

        c1, c2 = st.columns([8, 2])
        with c1:
            st.subheader("💬 AI Financial Assistant")
            st.markdown("Ask me to analyze stocks, search the news, or give suggestions based on technicals!")
        with c2:
            if st.button("🗑️ Clear Chat History", width="stretch"):
                # Archive the current thread and open a fresh one. Nothing is
                # deleted — the old conversation stays on disk and in the picker.
                current = chat_store.load(st.session_state.conversation_id)
                _open_conversation(chat_store.start_new(current))
                st.rerun()

        conversations = chat_store.list_conversations()
        if len(conversations) > 1:
            ids = [c.id for c in conversations]
            labels = {c.id: c.label() for c in conversations}
            try:
                position = ids.index(st.session_state.conversation_id)
            except ValueError:
                position = 0
            chosen = st.selectbox(
                "Conversation", ids, index=position,
                format_func=lambda i: labels.get(i, i), key="conversation_picker",
            )
            if chosen != st.session_state.conversation_id:
                selected = chat_store.load(chosen)
                if selected is not None:
                    chat_store.set_current(selected.id)
                    _open_conversation(selected)
                    st.rerun()

        st.caption(f"{len(st.session_state.messages)} messages · saved to `{CHAT_HISTORY_DIR}`")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("E.g., 'What is the latest news on NVDA and should I buy it?'"):
            # The agent needs the exchanges *before* this prompt, so capture the
            # history first and append afterwards.
            prior = list(st.session_state.messages)
            st.session_state.messages.append(make_message("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent is researching and thinking..."):
                    agent = get_financial_agent()
                    if not agent:
                        response = "⚠️ Please ensure AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_CREDENTIAL are configured in your .env file."
                    else:
                        try:
                            from langchain_core.messages import HumanMessage, AIMessage

                            formatted_history = [
                                HumanMessage(content=m["content"]) if m["role"] == "user"
                                else AIMessage(content=m["content"])
                                for m in prior
                            ]
                            result = agent.invoke({
                                "input": prompt,
                                "chat_history": formatted_history,
                            })
                            response = result["output"]
                        except Exception as e:
                            response = f"Agent encountered an error: {str(e)}"

                    st.markdown(response)

            st.session_state.messages.append(make_message("assistant", response))

            # Persist immediately so a crash or restart cannot lose the exchange.
            conversation = chat_store.load(st.session_state.conversation_id)
            if conversation is None:
                conversation = chat_store.create()
                st.session_state.conversation_id = conversation.id
            conversation.messages = st.session_state.messages
            chat_store.save(conversation)

if not auto_refresh_paused:
    time.sleep(refresh_rate)
    st.rerun()
else:
    st.sidebar.warning("⚠️ Auto-Refresh is paused. Uncheck the box in the sidebar to resume.")