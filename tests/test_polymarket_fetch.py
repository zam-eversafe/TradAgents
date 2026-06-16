"""Tests for Polymarket odds fetcher used by the Sentiment Analyst."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tradingagents.dataflows import polymarket


def _mock_event(title: str, slug: str = "test-event") -> dict:
    return {
        "id": "evt-1",
        "title": title,
        "slug": slug,
        "active": True,
        "closed": False,
        "volume1mo": 250000,
        "markets": [
            {
                "id": "mkt-1",
                "question": title,
                "active": True,
                "closed": False,
                "liquidity": "50000",
                "volume": "100000",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.72", "0.28"]',
                "oneWeekPriceChange": 0.05,
            }
        ],
    }


@pytest.mark.unit
class TestPolymarketHelpers:
    def test_parse_outcome_prices(self):
        market = {
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.60", "0.40"]',
        }
        assert polymarket._parse_outcome_prices(market) == [("Yes", 0.6), ("No", 0.4)]

    def test_passes_topic_filter_requires_informative_token(self):
        assert polymarket._passes_topic_filter("NVIDIA", "Will NVIDIA beat earnings?")
        assert not polymarket._passes_topic_filter("NVIDIA", "Will Apple beat earnings?")

    def test_build_search_queries_includes_company_name(self):
        queries = polymarket._build_search_queries("NVDA", "NVIDIA Corporation")
        assert queries[0] == "NVDA"
        assert "NVIDIA Corporation" in queries


@pytest.mark.unit
class TestFetchPolymarketOdds:
    def test_formats_relevant_markets(self):
        response = {"events": [_mock_event("Will NVIDIA beat Q2 earnings?")]}

        with patch.object(polymarket, "_search_gamma", return_value=response):
            block = polymarket.fetch_polymarket_odds("NVDA", company_name="NVIDIA")

        assert "Found 1 active Polymarket market" in block
        assert "Yes: 72.0%" in block
        assert "polymarket.com/event/test-event" in block

    def test_no_markets_returns_placeholder(self):
        with patch.object(polymarket, "_search_gamma", return_value={"events": []}):
            block = polymarket.fetch_polymarket_odds("ZZZZ")

        assert block.startswith("<no Polymarket markets found")

    def test_irrelevant_markets_filtered(self):
        response = {"events": [_mock_event("Will the Lakers win the NBA Finals?")]}

        with patch.object(polymarket, "_search_gamma", return_value=response):
            block = polymarket.fetch_polymarket_odds("NVDA", company_name="NVIDIA")

        assert block.startswith("<no relevant Polymarket markets")
