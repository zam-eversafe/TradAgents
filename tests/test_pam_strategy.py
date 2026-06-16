from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.utils.technical_indicators_tools import get_pam_strategy
from tradingagents.dataflows.y_finance import (
    analyze_price_action_manipulation_frame,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])


def _base_rows(days=24):
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        close = 100 + (idx % 3 - 1)
        rows.append((date.strftime("%Y-%m-%d"), close, 105, 95, close, 1_000_000))
    return rows


@pytest.mark.unit
class TestPamCoreDetections:
    def test_detects_bull_trap(self):
        rows = _base_rows()
        rows.append(("2026-01-25", 106, 109, 101, 103, 1_800_000))

        report = analyze_price_action_manipulation_frame(
            _frame(rows),
            symbol="MSFT",
            curr_date="2026-01-25",
            look_back_days=60,
            key_level_window=20,
        )

        assert "Bull Trap / Breakout Manipulation" in report
        assert "High pierced resistance" in report

    def test_detects_bear_trap_and_liquidity_grab(self):
        rows = _base_rows()
        rows.append(("2026-01-25", 96, 101, 91, 98, 2_000_000))

        report = analyze_price_action_manipulation_frame(
            _frame(rows),
            symbol="MSFT",
            curr_date="2026-01-25",
            look_back_days=60,
            key_level_window=20,
        )

        assert "Bear Trap / Breakdown Manipulation" in report
        assert "Stop Hunt / Liquidity Grab" in report
        assert "Pin Bar (hammer)" in report

    def test_detects_engulfing_confirmation(self):
        rows = _base_rows()
        rows[-1] = ("2026-01-24", 103, 104, 100, 101, 1_000_000)
        rows.append(("2026-01-25", 100, 105, 94, 104, 1_700_000))

        report = analyze_price_action_manipulation_frame(
            _frame(rows),
            symbol="MSFT",
            curr_date="2026-01-25",
            look_back_days=60,
            key_level_window=20,
        )

        assert "Mark Bar (bullish engulfing)" in report

    def test_reports_mother_baby_compression(self):
        rows = _base_rows(20)
        rows.extend(
            [
                ("2026-01-21", 100, 110, 90, 101, 1_000_000),
                ("2026-01-22", 101, 108, 93, 102, 900_000),
                ("2026-01-23", 102, 107, 94, 101, 900_000),
                ("2026-01-24", 101, 106, 95, 102, 900_000),
                ("2026-01-25", 102, 105, 96, 103, 900_000),
                ("2026-01-26", 103, 104, 97, 102, 900_000),
            ]
        )

        report = analyze_price_action_manipulation_frame(
            _frame(rows),
            symbol="MSFT",
            curr_date="2026-01-26",
            look_back_days=60,
            key_level_window=20,
        )

        assert "Mother/Baby price compression active" in report


@pytest.mark.unit
def test_pam_tool_wrapper_invokes_analyzer():
    with patch(
        "tradingagents.agents.utils.technical_indicators_tools.analyze_price_action_manipulation",
        return_value="PAM REPORT",
    ) as analyzer:
        result = get_pam_strategy.invoke(
            {
                "symbol": "MSFT",
                "curr_date": "2026-01-25",
                "look_back_days": 90,
                "key_level_window": 15,
            }
        )

    assert result == "PAM REPORT"
    analyzer.assert_called_once_with(
        symbol="MSFT",
        curr_date="2026-01-25",
        look_back_days=90,
        key_level_window=15,
    )


@pytest.mark.unit
def test_market_analyst_exposes_pam_tool():
    captured = {}

    class FakeLLM:
        def bind_tools(self, tools):
            captured["tools"] = tools
            return RunnableLambda(lambda _: SimpleNamespace(tool_calls=[], content="Technical report."))

    analyst = create_market_analyst(FakeLLM())
    result = analyst(
        {
            "trade_date": "2026-01-25",
            "company_of_interest": "MSFT",
            "messages": [],
        }
    )

    tool_names = {tool.name for tool in captured["tools"]}
    assert "get_pam_strategy" in tool_names
    assert result["market_report"] == "Technical report."
