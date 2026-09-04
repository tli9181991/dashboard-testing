# 🚀 AI-Powered Algorithmic Trading & Screening Dashboard

A fully modular, multi-asset trading dashboard built with Python, Streamlit, and LangChain. This system integrates real-time price monitoring, S&P 500 technical screening, AI-driven news sentiment analysis, and an interactive financial chatbot powered by Google's Gemini 1.5 Flash.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![Gemini AI](https://img.shields.io/badge/Gemini-1.5_Flash-orange)

## 🌟 Key Features

1. **📈 Portfolio Breakout Monitor**
   - Live tracking of crypto and stock assets via local `portfolio.csv`.
   - Automated breakout detection based on estimated Support/Resistance levels.
   - Dynamic exit rules (Trailing 10 SMA) with a `long_term` bypass flag for core holdings.
   - Interactive Plotly charts with candlestick data and 5/10/20/200 EMA overlays.

2. **🔍 S&P 500 Sector Screener**
   - Scans the entire S&P 500 for Stage-2 breakouts and high relative strength (vs SPY).
   - Automatically categorizes and displays the Top 5 performing stocks per sector.
   - Visualizes 6-month normalized relative performance charts.
   - Caches data locally to optimize API usage and speed.

3. **🎯 Swing Universe Funnel**
   - A seven-layer screen: liquidity gate → tradability score → regime & sector veto → setup layer → earnings gate → position sizing → composite ranking.
   - Two setups on daily bars: a momentum pullback continuation and an ICT 2022 model (sweep → displacement MSS → FVG entry).
   - Every candidate arrives as a complete trade plan — entry, stop, TP1, TP2, share count and dollar risk.
   - Runs offline against a synthetic universe, or against your portfolio, watchlist or a custom ticker list.
   - Also usable standalone: `python swing_screener.py --source demo`.

4. **🤖 Assistant**
   - Everything about one watchlist or portfolio symbol on a single page.
   - Price chart with 5/10/20/50 EMAs and labelled support/resistance — the same levels the breakout rule compares against, annotated with price and touch count.
   - Fundamentals across valuation, profitability, growth, balance sheet, dividend and analyst coverage, with missing fields shown as missing rather than zero.
   - News sentiment over a recent window (2 days by default). An empty window is reported as empty rather than scored.
   - Breakout and swing backtests for that symbol, with the simulation layer.
   - A chat that is handed the computed figures as context, so it reasons over the dashboard's own numbers rather than its recollection, and can search the web.

5. **🧠 AI Sector & News Sentiment**
   - Fetches the latest market news for your portfolio assets.
   - Leverages Google Gemini to synthesize news and generate an aggregated sentiment score and label (Bullish, Bearish, Neutral).

6. **💬 AI Financial Assistant (LangChain)**
   - A custom tool-calling agent equipped with `yfinance` and DuckDuckGo Web Search.
   - Capable of fetching live fundamentals, analyzing historical trends, and summarizing macroeconomic web news.
   - Features a robust conversational memory layer for continuous chat context.

## 🏗️ Project Structure

**Strategy engine** — one implementation, shared by the dashboard and the backtester:

- `strategy.py`: The signal rules. Pure and causal; every decision at bar *i* uses only bars `0..i`.
- `sizing.py`: Volatility-targeted position sizing.
- `regime.py`: Market regime gate on new entries.
- `backtest.py`: Event-driven replay, cost model, and performance metrics.
- `data.py`: Price loading with an on-disk cache, plus a deterministic synthetic generator for offline runs.
- `positions.py`: Reads `portfolio.csv` into explicit unit quantities.

**Application layer:**

- `app.py`: The Streamlit dashboard UI and orchestration layer.
- `breakout.py`: Live monitoring path — a thin shell over `strategy.evaluate`.
- `screening.py`: Scrapes the S&P 500, evaluates Stage-2 parameters, caches top sector performers.
- `fundamentals.py`: Structured fundamentals from yfinance, treating a missing field as missing.
- `assistant_charts.py`: Price chart with moving averages and labelled levels, drawn from the strategy's own level detection.
- `swing_screener.py`: The Swing Universe Funnel — liquidity, tradability, regime, setups, events, sizing and ranking. Runs in the dashboard or as a CLI.
- `chat_agent.py`: LangChain tool-calling agent with conversational memory.
- `sentiment.py`: AI-driven news scraper and sentiment evaluator.
- `config.py`: Environment variable loading and global parameters.
- `portfolio.csv`: Local state for active holdings.
- `tests/`: pytest suite, including the no-look-ahead proofs.

## ⚙️ Installation & Setup

```bash
git clone https://github.com/tli9181991/dashboard-testing.git
cd dashboard-testing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file for the AI features and risk defaults:

```
GOOGLE_API_KEY=your_key_here
ACCOUNT_EQUITY=100000
TARGET_VOL=0.15
MAX_POSITION_PCT=0.25
USE_REGIME_GATE=true
REGIME_BENCHMARK=^GSPC
```

Run the dashboard:

```bash
streamlit run app.py
```

## 🔬 Backtesting

The backtester replays bars forward and calls the same `strategy.evaluate` the live
dashboard calls, so the two cannot silently diverge.

```bash
python backtest.py --demo                        # synthetic data, no network needed
python backtest.py --symbols MRVL,NFLX --period 5y
python backtest.py --symbols AAPL --no-regime --show-trades
```

Output covers CAGR, annualised volatility, Sharpe, max drawdown, Calmar, win rate,
profit factor, expectancy per trade, time in market, and total costs paid.

`backtest.sweep()` varies one parameter at a time. Read it looking for a **plateau,
not a peak** — a setting that works at 10 but not at 9 or 11 is fitted noise and will
not survive live data.

### What makes the numbers trustworthy

Two properties, both covered by tests:

1. **Decide on the close, fill on the next open.** A signal from bar *t*'s close is
   only known once that bar completes, so orders fill at bar *t+1*'s open.
   `test_entries_fill_at_the_next_open_not_the_signal_close` asserts every fill price
   matches the *open* of its fill bar.
2. **Every fill pays.** Slippage moves price adversely and commission comes off the
   top. Breakout entries are exactly where real fills are worst, so a frictionless
   backtest of this strategy flatters it badly.

The causality contract is proved empirically in `tests/test_causality.py`: appending
future bars must never change a decision already taken. If those tests fail, every
number this repo produces is fiction.

## 🎲 Simulation

A single historical replay gives one number, and that number mixes the strategy,
the order the trades happened to arrive in, and the stretch of history it ran
through. `simulation.py` separates them:

- **Bootstrap** — resamples the realised trades thousands of times and rebuilds the
  equity curve, so you see the range of outcomes the same trades could have given.
  A realised path near the edge of the fan owed much of its result to sequence luck.
- **Random-entry benchmark** — replays the same number of trades with the same
  holding periods but random entry dates in the same names. If the strategy's
  average trade does not clear that distribution, the entry rule added nothing over
  simply being in the market that long. This is the sharpest of the three tests.
- **Buy and hold** — the honest floor. A strategy that only beats it by sitting in
  cash through a drawdown has not demonstrated selection skill.

The screener tab runs all three over the screened stocks. **Read the comparisons,
not the absolute return**: names selected today for having trended are not a fair
sample of what you could have picked back then, so absolute results from such a run
are inflated. The comparisons survive because every leg inherits the same bias.

### Backtesting the swing setups (tab 3)

The swing setups are brackets, not a moving-average rule, so they get their own
engine — `swing_backtest.py`, a triple-barrier replay:

- Each setup's entry is placed as a **resting order** — a stop-buy above the market
  for the momentum pullback, a limit-buy below it for the ICT setup — and cancelled
  if it never trades. Typically half of all setups never become positions, and an
  engine that assumes they all fill will report an edge that does not exist.
- **Gap fills are modelled.** A stop-buy that gaps through its trigger fills at the
  open, worse than the order; a limit-buy that gaps below fills better.
- **TP1 takes a partial and moves the stop to breakeven**, as the setup's own design
  intends; the remainder runs to TP2 or the time stop.
- **The intrabar unknown is reported as a range.** When one daily bar covers both the
  stop and a target, OHLC cannot say which came first, so the tab runs both
  resolutions. A wide gap between them means daily bars cannot settle the strategy
  and it needs intraday data before it is traded.
- **MAE and MFE are recorded in R** for every trade, which is what tells you whether
  the stops sit where the trades actually need them.

R multiples *add* rather than compound — every trade is sized to the same risk — so
the resampled fan for these setups is a running sum, not a product.

## 🛡️ Risk Controls

**Volatility-targeted sizing** replaces the old fixed six-share position:

```
target_notional = equity × target_vol / annualised_vol(symbol)
```

capped by `MAX_POSITION_PCT`. Each position then contributes a comparable amount of
*risk* rather than a comparable number of shares — six shares of a $30 utility and six
of a $300 semiconductor were never the same bet.

**Regime gate** blocks new entries while the benchmark trades below its 200 SMA. Exits
are never gated: a risk-off reading must not trap an open position.

## 🧪 Tests

```bash
pytest tests/ -q
```

## ⚠️ Status

This is a research and monitoring tool. It generates signals; it does **not** place
orders, and there is no broker integration, order ledger, or reconciliation layer.
Before trading this live you would need, at minimum: a headless execution daemon
(Streamlit reruns its whole script on every interaction, which is not a safe place to
put order placement), an order lifecycle with idempotency keys, broker-state
reconciliation on startup, and a data source more reliable than yfinance — which is
unofficial, returns empty frames when throttled, and restates history after splits.

Measure expectancy in the backtester before risking capital, and treat "no edge" as a
valid and useful finding.
