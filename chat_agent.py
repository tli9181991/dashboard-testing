import os
import yfinance as yf
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from config import AZURE_INFERENCE_ENDPOINT, AZURE_INFERENCE_CREDENTIAL, DEEPSEEK_MODEL_NAME

def get_financial_agent():
    """Initializes a native LangChain tool-calling agent with memory, backed by
    DeepSeek V4 Flash deployed in Azure AI Foundry."""
    if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_CREDENTIAL:
        return None

    @tool
    def get_stock_fundamentals(ticker: str) -> str:
        """Use this to get the current price, P/E ratio, and analyst recommendations for a specific stock ticker."""
        try:
            info = yf.Ticker(ticker).info
            price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
            fwd_pe = info.get("forwardPE", "N/A")
            recommendation = info.get("recommendationKey", "N/A")
            return f"Ticker: {ticker} | Price: ${price} | Forward P/E: {fwd_pe} | Analyst Consensus: {recommendation}"
        except Exception as e:
            return f"Failed to fetch fundamentals for {ticker}: {str(e)}"

    @tool
    def get_historical_performance(ticker: str) -> str:
        """Use this to get the 6-month percentage return and price trend for a specific stock ticker."""
        try:
            hist = yf.Ticker(ticker).history(period="6mo")
            if hist.empty:
                return "No historical data found."
            start_px = float(hist['Close'].iloc[0])
            end_px = float(hist['Close'].iloc[-1])
            ret = ((end_px - start_px) / start_px) * 100
            return f"{ticker} 6-month return: {ret:.2f}%. Current price is ${end_px:.2f}."
        except Exception as e:
            return f"Error fetching history for {ticker}: {str(e)}"

    search_tool = DuckDuckGoSearchRun()
    
    @tool
    def web_news_search(query: str) -> str:
        """Use this to search the web for the latest news, market events, or macroeconomic updates regarding a stock."""
        return search_tool.run(query)

    # Bind tools directly to the DeepSeek V4 Flash LLM (via Azure AI Foundry)
    tools = [get_stock_fundamentals, get_historical_performance, web_news_search]
    llm = AzureAIChatCompletionsModel(
        endpoint=AZURE_INFERENCE_ENDPOINT,
        credential=AZURE_INFERENCE_CREDENTIAL,
        model=DEEPSEEK_MODEL_NAME,
        temperature=0.1,
    ).bind_tools(tools)
    tools_map = {t.name: t for t in tools}

    class BulletproofAgent:
        def invoke(self, inputs):
            # 1. Setup Conversation Context
            messages = [SystemMessage(content="You are a helpful financial assistant. Use tools to answer questions. Always rely on the tools for up-to-date information.")] 
            messages += inputs.get("chat_history", [])
            messages.append(HumanMessage(content=inputs["input"]))
            
            # 2. Initial LLM call
            response = llm.invoke(messages)
            
            # 3. Tool Execution Loop
            while response.tool_calls:
                messages.append(response)
                for tool_call in response.tool_calls:
                    selected_tool = tools_map[tool_call["name"]]
                    tool_msg = selected_tool.invoke(tool_call["args"])
                    messages.append(ToolMessage(content=str(tool_msg), tool_call_id=tool_call["id"]))
                
                # Send tool results back to LLM for final answer
                response = llm.invoke(messages)
                
            return {"output": response.content}

    return BulletproofAgent()