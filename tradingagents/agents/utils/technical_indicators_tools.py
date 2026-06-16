from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.y_finance import analyze_price_action_manipulation
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)


@tool
def get_pam_strategy(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many calendar days to look back for PAM events"] = 120,
    key_level_window: Annotated[int, "prior trading bars used to define support/resistance"] = 20,
) -> str:
    """
    Analyze Price Action Manipulation (PAM) setups for a ticker.

    PAM looks for traps, stop hunts, liquidity grabs, failed breakouts,
    pump-and-dump behavior, distribution shakeouts, and UC1/DC1 continuation
    setups using OHLCV price action around recent support/resistance levels.
    """
    return analyze_price_action_manipulation(
        symbol=symbol,
        curr_date=curr_date,
        look_back_days=look_back_days,
        key_level_window=key_level_window,
    )
