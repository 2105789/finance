import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from alpha_vantage.timeseries import TimeSeries
from core.config import TAVILY_API_KEY, ALPHA_VANTAGE_API_KEY

def get_tavily_search_tool():
    """Initializes and returns the Tavily search tool."""
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    search = TavilySearchResults(max_results=3) # Keep it concise for agent use
    return Tool(
        name="TavilySearch",
        func=search.run,
        description="Useful for when you need to answer questions about current events, data, or facts. Input should be a search query."
    )

def get_stock_price(symbol: str):
    """Fetches the latest stock price for a given symbol using Alpha Vantage."""
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='json')
        data, meta_data = ts.get_quote_endpoint(symbol=symbol)
        if data and '05. price' in data:
            return f"The current price of {symbol} is {data['05. price']}."
        else:
            return f"Could not retrieve stock price for {symbol}. Response: {data}"
    except Exception as e:
        return f"Error fetching stock price for {symbol}: {e}"

def get_alpha_vantage_tool():
    """Initializes and returns the Alpha Vantage stock price tool."""
    return Tool(
        name="AlphaVantageStockPrice",
        func=get_stock_price,
        description="Useful for when you need to get the current stock price for a specific company symbol. Input should be the stock symbol (e.g., 'MSFT')."
    )

def get_tools():
    """Returns a list of all available tools."""
    return [get_tavily_search_tool(), get_alpha_vantage_tool()] 