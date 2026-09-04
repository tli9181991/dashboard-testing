import os
import json
import yfinance as yf

from datetime import datetime, timedelta, timezone
from typing import Literal
from pydantic import BaseModel, Field
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

from config import AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_CREDENTIAL, DEEPSEEK_MODEL_NAME

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

def get_hourly_sentiment(ticker: str) -> dict:
    if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_CREDENTIAL:
        return {"error": "Missing Azure AI Foundry endpoint/credential (AZURE_INFERENCE_ENDPOINT / AZURE_INFERENCE_CREDENTIAL)."}

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

    llm = AzureAIChatCompletionsModel(
        endpoint=AZURE_INFERENCE_ENDPOINT,
        credential=AZURE_INFERENCE_CREDENTIAL,
        model=DEEPSEEK_MODEL_NAME,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(SentimentResult)
    payload = [item.model_dump() for item in news_list]
    
    try:
        res = structured_llm.invoke([("system", SENTIMENT_SYSTEM), ("user", json.dumps({"ticker": ticker, "news": payload}, default=str))])
        return {"data": res.model_dump(), "articles": payload} 
    except Exception as e:
        return {"error": f"LLM Error: {str(e)}"}

def _published_at(raw: dict) -> datetime | None:
    """Publication time from whichever shape yfinance returned this call."""
    content = raw.get("content", raw)
    stamp = content.get("pubDate") or raw.get("providerPublishTime")
    if stamp is None:
        return None
    if isinstance(stamp, (int, float)):
        return datetime.fromtimestamp(stamp, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def collect_recent_news(ticker: str, days: int = 2, max_items: int = 12) -> dict:
    """Headlines published within the last ``days``, newest first.

    Returns the articles kept, how many were dropped as stale, and the window used.
    Filtering happens before the model is called: scoring last month's headlines as
    though they were today's is worse than reporting that nothing has been published.
    """
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception as exc:
        return {"articles": [], "fetched": 0, "stale": 0, "days": days,
                "error": f"News lookup failed: {exc}"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    kept, stale = [], 0

    for raw in raw_items:
        published = _published_at(raw)
        if published is None or published < cutoff:
            stale += 1
            continue
        content = raw.get("content", raw)
        kept.append(NewsItem(
            ticker=ticker,
            title=str(content.get("title") or "Untitled"),
            publisher=(content.get("provider") or {}).get("displayName") or raw.get("publisher"),
            published_at=published.isoformat(),
            summary=content.get("summary") or content.get("description"),
        ))

    kept.sort(key=lambda item: item.published_at or "", reverse=True)
    return {"articles": [item.model_dump() for item in kept[:max_items]],
            "fetched": len(raw_items), "stale": stale, "days": days, "error": ""}


def get_recent_sentiment(ticker: str, days: int = 2, max_items: int = 12) -> dict:
    """Sentiment over the last ``days`` of news only.

    An empty window is reported as such rather than scored — "no news in two days"
    is a real and useful answer, and manufacturing a neutral reading from nothing
    would look identical to a genuine neutral reading on real coverage.
    """
    if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_CREDENTIAL:
        return {"error": "Missing Azure AI Foundry endpoint/credential "
                         "(AZURE_INFERENCE_ENDPOINT / AZURE_INFERENCE_CREDENTIAL)."}

    news = collect_recent_news(ticker, days=days, max_items=max_items)
    if news["error"]:
        return {"error": news["error"], "window": news}
    if not news["articles"]:
        return {"error": f"No {ticker} headlines published in the last {days} days "
                         f"({news['stale']} older stories were skipped).",
                "window": news}

    llm = AzureAIChatCompletionsModel(
        endpoint=AZURE_INFERENCE_ENDPOINT,
        credential=AZURE_INFERENCE_CREDENTIAL,
        model=DEEPSEEK_MODEL_NAME,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(SentimentResult)
    prompt = json.dumps({"ticker": ticker, "window_days": days,
                         "news": news["articles"]}, default=str)

    try:
        res = structured_llm.invoke([("system", SENTIMENT_SYSTEM), ("user", prompt)])
        return {"data": res.model_dump(), "articles": news["articles"], "window": news}
    except Exception as exc:
        return {"error": f"LLM Error: {exc}", "window": news}


def sentiment_prompt_text(payload: dict) -> str:
    """Render a sentiment result as plain text for the assistant's context."""
    window = payload.get("window") or {}
    days = window.get("days", 2)
    if payload.get("error"):
        return f"News sentiment (last {days} days): {payload['error']}"

    data = payload.get("data", {})
    lines = [
        f"News sentiment over the last {days} days: {data.get('label', 'unknown')} "
        f"(score {data.get('score', 0):+.2f}, confidence {data.get('confidence', 0):.0%}) "
        f"from {len(payload.get('articles', []))} headlines."
    ]
    for key, heading in (("positive_factors", "Positive"), ("negative_factors", "Negative")):
        items = data.get(key) or []
        if items:
            lines.append(f"{heading}: " + "; ".join(str(i) for i in items))
    for article in payload.get("articles", [])[:6]:
        lines.append(f"- [{article.get('publisher') or 'unknown'}] {article.get('title')}")
    return "\n".join(lines)


def analyze_sector_with_deepseek(sector_name: str, etf_ticker: str, top_stocks: list[str]) -> SectorAnalysisResult:
    llm = AzureAIChatCompletionsModel(
        endpoint=AZURE_INFERENCE_ENDPOINT,
        credential=AZURE_INFERENCE_CREDENTIAL,
        model=DEEPSEEK_MODEL_NAME,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(SectorAnalysisResult)
    
    prompt = f"Sector: {sector_name} ({etf_ticker})\nTop Stocks: {top_stocks}"
    return structured_llm.invoke([("system", "Summarize sector trend and sentiment."), ("user", prompt)])