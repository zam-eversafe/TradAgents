import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config
from .utils import safe_ticker_component
from .symbol_utils import normalize_symbol, NoMarketDataError

logger = logging.getLogger(__name__)


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the date column to ``Date``.

    Some yfinance builds leave the index unnamed (so ``reset_index()`` yields
    ``index``) or use ``Datetime`` for intraday data. Rename the first
    date-like column so indicators don't silently drop when it isn't ``Date``.
    """
    if "Date" in data.columns:
        return data
    for candidate in ("index", "Datetime", "date"):
        if candidate in data.columns:
            return data.rename(columns={candidate: "Date"})
    return data


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data = _ensure_date_column(data)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def _max_ohlcv_date(data: pd.DataFrame) -> pd.Timestamp | None:
    """Return the latest ``Date`` with a usable ``Close``, or ``None`` if empty.

    Yahoo sometimes appends calendar rows with volume but NaN OHLC. Those rows
    are dropped by ``_clean_dataframe`` and must not count as fresh cache data.
    """
    if data is None or data.empty:
        return None
    frame = _ensure_date_column(data).copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if "Close" in frame.columns:
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"])
    else:
        frame = frame.dropna(subset=["Date"])
    if frame.empty:
        return None
    return frame["Date"].max().normalize()


def _expected_latest_cache_date(
    curr_date_dt: pd.Timestamp, today_date: pd.Timestamp
) -> pd.Timestamp:
    """Latest row Yahoo should have when downloading with ``end=today``.

    yfinance treats ``end`` as exclusive, so the newest daily bar is usually
    the prior session. Cap at ``curr_date`` so historical backtests do not
    force a refresh when the cache already covers the requested window.
    """
    latest_available = (today_date.normalize() - pd.offsets.BDay(1)).normalize()
    return min(curr_date_dt.normalize(), latest_available)


def expected_latest_trading_session(
    curr_date: str, *, as_of: pd.Timestamp | None = None
) -> pd.Timestamp:
    """Last session expected to have a ``Close`` on or before ``curr_date``."""
    ref = (as_of or pd.Timestamp.today()).normalize()
    return _expected_latest_cache_date(pd.to_datetime(curr_date), ref)


def format_missing_session_label(
    date_str: str,
    *,
    curr_date: str,
    latest_session: pd.Timestamp | None,
) -> str:
    """Explain why a calendar date has no indicator/price row in tool output."""
    missing = pd.to_datetime(date_str).normalize()
    curr = pd.to_datetime(curr_date).normalize()
    today = pd.Timestamp.today().normalize()
    latest = latest_session.normalize() if latest_session is not None else None

    if missing.weekday() >= 5:
        return "N/A: Not a trading day (weekend or holiday)"

    if (
        latest is not None
        and missing > latest
        and missing == curr
        and missing >= today
    ):
        return (
            "N/A: Session not yet available "
            "(analysis date is today; no complete daily bar before market close)"
        )

    if latest is not None and missing > latest:
        return "N/A: No market data for this date"

    return "N/A: Not a trading day (weekend or holiday)"


def _yahoo_download_end_date(today_date: pd.Timestamp) -> str:
    """Exclusive Yahoo ``end`` bound that includes the latest daily bar."""
    return (today_date.normalize() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")


def _download_yahoo_ohlcv(canonical: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo and normalize to a ``Date`` column."""
    downloaded = yf_retry(
        lambda: yf.download(
            canonical,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        )
    )
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    return _ensure_date_column(downloaded.reset_index())


def _merge_ohlcv_frames(base: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    """Merge two OHLCV frames, preferring rows with a usable ``Close`` per date."""
    if base is None or base.empty:
        return tail.copy() if tail is not None else pd.DataFrame()
    if tail is None or tail.empty:
        return base.copy()

    frames = []
    for frame in (base, tail):
        part = _ensure_date_column(frame.copy())
        part["Date"] = pd.to_datetime(part["Date"], errors="coerce").dt.normalize()
        if "Close" in part.columns:
            part["Close"] = pd.to_numeric(part["Close"], errors="coerce")
        frames.append(part.dropna(subset=["Date"]))

    combined = pd.concat(frames, ignore_index=True)
    combined["_has_close"] = combined["Close"].notna()
    combined = combined.sort_values(["Date", "_has_close"])
    return combined.drop_duplicates(subset=["Date"], keep="last").drop(
        columns="_has_close"
    ).reset_index(drop=True)


def _patch_incomplete_tail(
    data: pd.DataFrame,
    canonical: str,
    curr_date_dt: pd.Timestamp,
    today_date: pd.Timestamp,
) -> pd.DataFrame:
    """Backfill recent sessions when bulk Yahoo history has NaN OHLC rows.

    Yahoo often returns the latest one or two daily bars with NaN ``Close``
    in long-window ``yf.download`` calls while a short-window fetch for the
    same dates returns complete OHLCV. When the latest usable row trails the
    expected session, fetch from the day after that row and merge.
    """
    if data is None or data.empty:
        return data

    max_valid = _max_ohlcv_date(data)
    expected = _expected_latest_cache_date(curr_date_dt, today_date)
    if max_valid is not None and max_valid >= expected:
        return data

    tail_start_dt = (
        (max_valid + pd.Timedelta(days=1)).normalize()
        if max_valid is not None
        else (expected - pd.DateOffset(days=60)).normalize()
    )
    tail_start = tail_start_dt.strftime("%Y-%m-%d")
    end_str = _yahoo_download_end_date(today_date)
    if tail_start >= end_str:
        return data

    tail = _download_yahoo_ohlcv(canonical, tail_start, end_str)
    if tail.empty:
        logger.warning(
            "Tail OHLCV refresh for %s returned no rows (%s to %s)",
            canonical,
            tail_start,
            end_str,
        )
        return data

    merged = _merge_ohlcv_frames(data, tail)
    patched_max = _max_ohlcv_date(merged)
    if patched_max is not None and (max_valid is None or patched_max > max_valid):
        logger.info(
            "Patched incomplete OHLCV tail for %s (%s -> %s)",
            canonical,
            max_valid.strftime("%Y-%m-%d") if max_valid is not None else "none",
            patched_max.strftime("%Y-%m-%d"),
        )
    return merged


def _is_cache_stale(
    data: pd.DataFrame,
    curr_date_dt: pd.Timestamp,
    today_date: pd.Timestamp,
) -> bool:
    max_date = _max_ohlcv_date(data)
    if max_date is None:
        return True
    return max_date < _expected_latest_cache_date(curr_date_dt, today_date)


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads five years of data up to today and caches per symbol per day.
    A cache hit is refreshed when its latest row is behind the most recent
    trading day Yahoo should expose for the requested analysis date. Rows
    after curr_date are filtered out so backtests never see future prices.
    """
    # Resolve broker/forex symbols (XAUUSD+ -> GC=F) to Yahoo's convention,
    # then reject values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache uses a fixed window (5y to today) so one file per symbol per day.
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")
    download_end_str = _yahoo_download_end_date(today_date)

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    # A cached file may be empty if a prior fetch failed (unknown symbol,
    # transient rate limit). Treat an empty/columnless cache as a miss and
    # re-fetch rather than serving the poisoned file forever.
    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            if _is_cache_stale(cached, curr_date_dt, today_date):
                expected = _expected_latest_cache_date(curr_date_dt, today_date)
                latest = _max_ohlcv_date(cached)
                logger.info(
                    "OHLCV cache for %s is stale (latest row %s, expected at least %s); re-fetching",
                    canonical,
                    latest.strftime("%Y-%m-%d") if latest is not None else "unknown",
                    expected.strftime("%Y-%m-%d"),
                )
            else:
                data = cached

    if data is None:
        downloaded = _download_yahoo_ohlcv(canonical, start_str, download_end_str)
        # Only cache real data — never persist an empty frame.
        if downloaded.empty or "Close" not in downloaded.columns:
            raise NoMarketDataError(
                symbol, canonical, "Yahoo Finance returned no rows"
            )
        downloaded = _patch_incomplete_tail(
            downloaded, canonical, curr_date_dt, today_date
        )
        downloaded.to_csv(data_file, index=False, encoding="utf-8")
        data = downloaded
    else:
        patched = _patch_incomplete_tail(data, canonical, curr_date_dt, today_date)
        if _max_ohlcv_date(patched) != _max_ohlcv_date(data):
            patched.to_csv(data_file, index=False, encoding="utf-8")
        data = patched

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt]

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
        latest_session = (
            pd.to_datetime(data.iloc[-1]["Date"]) if not data.empty else None
        )

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        return format_missing_session_label(
            curr_date_str,
            curr_date=curr_date,
            latest_session=latest_session,
        )
