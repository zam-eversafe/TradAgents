"""Polymarket prediction-market vendor.

Surfaces live, market-implied probabilities for forward-looking events (Fed
decisions, recession, elections, geopolitics, crypto) to the news analyst, as a
complement to news (what happened) and FRED macro data (where things stand):
what the crowd actually prices to happen next.

Also provides ticker-specific odds for the Sentiment Analyst via
``fetch_polymarket_odds``.

Uses Polymarket's public Gamma API (https://gamma-api.polymarket.com) — no key,
no auth. Each market's ``outcomePrices`` are the implied probabilities of its
outcomes (a "Yes" at 0.76 means the market prices a 76% chance).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import requests

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

# Default number of markets to return, ranked by traded volume.
DEFAULT_LIMIT = 6


def _request(path: str, params: dict) -> dict:
    response = requests.get(
        f"{GAMMA_BASE}/{path}", params=params, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _parse_json_list(value) -> list:
    """Gamma encodes ``outcomes``/``outcomePrices`` as JSON-string arrays."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _is_forward_looking(market: dict, now: datetime) -> bool:
    """Keep only open markets that resolve in the future.

    ``closed`` is the reliable resolved flag (``active`` stays True even for
    settled markets), and a past ``endDate`` means the event already resolved —
    either way it is not a forward-looking signal.
    """
    if market.get("closed"):
        return False
    end_date = market.get("endDate")
    if end_date:
        try:
            if datetime.fromisoformat(end_date.replace("Z", "+00:00")) < now:
                return False
        except ValueError:
            pass
    return bool(_parse_json_list(market.get("outcomePrices"))) and bool(
        _parse_json_list(market.get("outcomes"))
    )


def get_prediction_markets(topic: str, limit: int | None = None) -> str:
    """Return live prediction-market probabilities for an event topic.

    Args:
        topic: Event keyword(s), e.g. "Fed rate cut", "recession 2026",
            "US election", or a sector/company event.
        limit: Max markets to return (ranked by traded volume); ``None`` uses
            DEFAULT_LIMIT.

    Returns:
        A markdown report of the most-traded open markets matching the topic,
        each with its implied probability, traded volume, resolution date, and
        recent (1-week) move.
    """
    if limit is None:
        limit = DEFAULT_LIMIT

    try:
        data = _request("public-search", {"q": topic, "limit_per_type": 20})
    except requests.RequestException as e:
        logger.warning("Polymarket search failed for %r: %s", topic, e)
        return (
            f"Polymarket data is currently unavailable (network error: {e}). "
            f"Proceed without prediction-market signal for '{topic}'."
        )

    now = datetime.now(timezone.utc)
    candidates = [
        m
        for event in data.get("events", [])
        for m in event.get("markets", [])
        if _is_forward_looking(m, now)
    ]
    candidates.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)

    header = (
        f'## Polymarket prediction markets: "{topic}"\n'
        f"Live, market-implied probabilities (higher traded volume = deeper, "
        f"more reliable). A probability is the crowd's priced odds of the event, "
        f"not a forecast you should take as certain.\n\n"
    )

    if not candidates:
        return header + (
            f"No open prediction markets matched '{topic}'. Polymarket coverage "
            f"is concentrated in macro, political, geopolitical, and crypto "
            f"events; a specific equity may have none."
        )

    lines = []
    for m in candidates[:limit]:
        prices = _parse_json_list(m.get("outcomePrices"))
        outcomes = _parse_json_list(m.get("outcomes"))
        try:
            prob = float(prices[0])
        except (ValueError, IndexError):
            continue
        label = outcomes[0] if outcomes else "Yes"
        volume = m.get("volumeNum") or 0
        end_date = (m.get("endDate") or "")[:10]
        wk = m.get("oneWeekPriceChange")
        wk_str = (
            f", 1-week {wk * 100:+.1f}pp"
            if isinstance(wk, (int, float)) and wk
            else ""
        )
        lines.append(
            f"- **{m.get('question')}** — {label} {prob:.0%} "
            f"(${volume:,.0f} volume, resolves {end_date}{wk_str})"
        )

    return header + "\n".join(lines) + "\n"


_GAMMA_SEARCH_URL = f"{GAMMA_BASE}/public-search"
_UA = "tradingagents/0.3 (+https://github.com/TauricResearch/TradingAgents)"
_MAX_EVENTS = 8
_SEARCH_PAGES = 2

_NOISE_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "of", "for", "and", "or", "to", "is", "are",
    "was", "were", "will", "be", "by", "with", "from", "as", "it", "its", "not", "no",
    "inc", "corp", "ltd", "plc", "co", "company", "group", "holdings",
    "market", "odds", "prediction", "forecast", "chance", "probability",
})


def _search_gamma(query: str, page: int = 1, timeout: float = 12.0) -> dict[str, Any]:
    params = {
        "q": query,
        "page": str(page),
        "events_status": "active",
        "keep_closed_markets": "0",
    }
    url = f"{_GAMMA_SEARCH_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Polymarket search failed for %r page %s: %s", query, page, exc)
        return {"events": [], "error": str(exc)}


def _parse_outcome_prices(market: dict[str, Any]) -> list[tuple[str, float]]:
    outcomes_raw = market.get("outcomes") or []
    prices_raw = market.get("outcomePrices")
    if not prices_raw:
        return []
    try:
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except (json.JSONDecodeError, TypeError):
        return []
    result: list[tuple[str, float]] = []
    for i, price in enumerate(prices):
        try:
            p = float(price)
        except (ValueError, TypeError):
            continue
        name = outcomes[i] if i < len(outcomes) else f"Outcome {i + 1}"
        result.append((name, p))
    return result


def _format_price_movement(market: dict[str, Any]) -> Optional[str]:
    changes = [
        (abs(market.get("oneDayPriceChange") or 0), market.get("oneDayPriceChange"), "today"),
        (abs(market.get("oneWeekPriceChange") or 0), market.get("oneWeekPriceChange"), "this week"),
        (abs(market.get("oneMonthPriceChange") or 0), market.get("oneMonthPriceChange"), "this month"),
    ]
    changes.sort(key=lambda x: x[0], reverse=True)
    abs_change, raw_change, period = changes[0]
    if abs_change < 0.01:
        return None
    direction = "up" if raw_change > 0 else "down"
    return f"{direction} {abs_change * 100:.1f}% {period}"


def _passes_topic_filter(topic: str, title: str) -> bool:
    """Require at least one informative topic token in the event title."""
    topic_tokens = [
        w for w in re.sub(r"[^\w\s]", " ", topic.lower()).split()
        if len(w) > 1 and w not in _NOISE_WORDS
    ]
    if not topic_tokens:
        return True
    title_lower = " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())
    title_words = set(title_lower.split())
    for token in topic_tokens:
        if token in title_words:
            return True
        if len(token) >= 4 and token in title_lower:
            return True
    return False


def _build_search_queries(ticker: str, company_name: Optional[str]) -> list[str]:
    queries = [ticker.upper()]
    if company_name:
        cleaned = re.sub(
            r"\b(inc\.?|corp\.?|ltd\.?|plc|co\.?|company|group|holdings)\b",
            "",
            company_name,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned and cleaned.lower() not in {ticker.lower(), ticker.upper()}:
            queries.append(cleaned)
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    return unique


def _collect_events(queries: list[str]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for query in queries:
        for page in range(1, _SEARCH_PAGES + 1):
            response = _search_gamma(query, page=page)
            for event in response.get("events", []):
                event_id = event.get("id")
                if event_id:
                    by_id[event_id] = event
    return list(by_id.values())


def _extract_market_items(events: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for event in events:
        if event.get("closed") or not event.get("active", True):
            continue
        title = event.get("title", "")
        if topic and not _passes_topic_filter(topic, title):
            continue

        markets = event.get("markets") or []
        active_markets = []
        for m in markets:
            if m.get("closed") or not m.get("active", True):
                continue
            try:
                liq = float(m.get("liquidity", 0) or 0)
            except (ValueError, TypeError):
                liq = 0
            if liq > 0:
                active_markets.append(m)
        if not active_markets:
            continue

        active_markets.sort(
            key=lambda m: float(m.get("volume", 0) or 0),
            reverse=True,
        )
        top_market = active_markets[0]
        outcome_prices = _parse_outcome_prices(top_market)
        if not outcome_prices:
            continue

        slug = event.get("slug", "")
        event_id = event.get("id", "")
        url = (
            f"https://polymarket.com/event/{slug}"
            if slug
            else f"https://polymarket.com/event/{event_id}"
        )
        items.append({
            "title": title,
            "question": top_market.get("question", title),
            "url": url,
            "outcome_prices": outcome_prices[:3],
            "movement": _format_price_movement(top_market),
            "volume": float(event.get("volume1mo") or event.get("volume24hr") or 0),
        })

    items.sort(key=lambda x: x["volume"], reverse=True)
    return items[:_MAX_EVENTS]


def _format_items(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        odds_parts = [f"{name}: {price * 100:.1f}%" for name, price in item["outcome_prices"]]
        odds_str = ", ".join(odds_parts)
        movement = f" ({item['movement']})" if item.get("movement") else ""
        lines.append(
            f"- [{item['title']}]({item['url']}) — {odds_str}{movement}"
        )
    return "\n".join(lines)


def fetch_polymarket_odds(
    ticker: str,
    company_name: Optional[str] = None,
) -> str:
    """Fetch relevant Polymarket odds for ``ticker`` and return a prompt block.

    Searches by ticker and optional ``company_name`` (from yfinance identity).
    Returns a placeholder string on failure or when no relevant markets exist.
    """
    queries = _build_search_queries(ticker, company_name)
    events = _collect_events(queries)
    if not events:
        return f"<no Polymarket markets found for {ticker.upper()}>"

    topic = company_name or ticker
    items = _extract_market_items(events, topic)
    if not items and company_name and company_name.lower() != ticker.lower():
        items = _extract_market_items(events, ticker)

    if not items:
        return f"<no relevant Polymarket markets found for {ticker.upper()}>"

    header = (
        f"Found {len(items)} active Polymarket market(s) related to "
        f"{ticker.upper()}"
    )
    if company_name:
        header += f" / {company_name}"
    return header + ":\n\n" + _format_items(items)
