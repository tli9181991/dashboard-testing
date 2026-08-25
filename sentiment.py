import os
import json
import yfinance as yf

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class NewsItem(BaseModel):
    ticker: str
    title: str
    publisher: str | None = None
    published_at: str | None = None
    summary: str | None = None

class SentimentResult(BaseModel):
    ticker: str
    label: Literal["positive", "neutral", "negative", "mixed", "insufficient"]
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)

class SectorAnalysisResult(BaseModel):
    sector_name: str
    sector_etf: str
    trend_description: str
    sentiment_label: str
    sentiment_summary: str

SENTIMENT_SYSTEM = "You are a financial-news classification AI. Classify the news. Return structured JSON."

def get_hourly_sentiment(ticker: str, api_key: str) -> dict:
    if not api_key: return {"error": "Missing Gemini API Key."}
    os.environ["GOOGLE_API_KEY"] = api_key
    
    try: raw_items = yf.Ticker(ticker).news or []
    except Exception: raw_items = []
        
    news_list = []
    for raw in raw_items[:6]:
        content = raw.get("content", raw)
        published = content.get("pubDate") or raw.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
        
        news_list.append(NewsItem(
            ticker=ticker, title=str(content.get("title") or "Untitled"),
            publisher=content.get("provider", {}).get("displayName") or raw.get("publisher"),
            published_at=str(published) if published else None,
            summary=content.get("summary") or content.get("description")
        ))
        
    if not news_list: return {"error": "No recent news found."}
        
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(SentimentResult)
    payload = [item.model_dump() for item in news_list]
    
    try:
        res = structured_llm.invoke([("system", SENTIMENT_SYSTEM), ("user", json.dumps({"ticker": ticker, "news": payload}, default=str))])
        return {"data": res.model_dump(), "articles": payload} 
    except Exception as e:
        return {"error": f"LLM Error: {str(e)}"}

def analyze_sector_with_gemini(sector_name: str, etf_ticker: str, top_stocks: list[str], api_key: str) -> SectorAnalysisResult:
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(SectorAnalysisResult)
    
    prompt = f"Sector: {sector_name} ({etf_ticker})\nTop Stocks: {top_stocks}"
    return structured_llm.invoke([("system", "Summarize sector trend and sentiment."), ("user", prompt)])