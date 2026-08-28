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

3. **🧠 AI Sector & News Sentiment**
   - Fetches the latest market news for your portfolio assets.
   - Leverages Google Gemini to synthesize news and generate an aggregated sentiment score and label (Bullish, Bearish, Neutral).

4. **💬 AI Financial Assistant (LangChain)**
   - A custom tool-calling agent equipped with `yfinance` and DuckDuckGo Web Search.
   - Capable of fetching live fundamentals, analyzing historical trends, and summarizing macroeconomic web news.
   - Features a robust conversational memory layer for continuous chat context.

## 🏗️ Project Structure

- `app.py`: The main Streamlit dashboard UI and orchestration layer.
- `breakout.py`: Logic for technical indicators, moving averages, and breakout/exit signals.
- `screening.py`: Logic for scraping the S&P 500, evaluating Stage-2 parameters, and caching top sector performers.
- `chat_agent.py`: LangChain native tool-calling agent with conversational memory.
- `sentiment.py`: AI-driven news scraper and sentiment analysis evaluator.
- `config.py`: Environment variable loading and global application parameters.
- `portfolio.csv`: Local state-management database for active holdings.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ai-trading-dashboard.git](https://github.com/yourusername/ai-trading-dashboard.git)
   cd ai-trading-dashboard