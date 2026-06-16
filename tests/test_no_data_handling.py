"""Tests that empty vendor results never become fabricated data.

Covers two systematic fixes:
  - load_ohlcv must not cache an empty download (cache poisoning), and must
    raise NoMarketDataError instead of returning an empty frame.
  - route_to_vendor must convert NoMarketDataError into a single explicit
    "NO_DATA_AVAILABLE" sentinel after all vendors are exhausted.
"""

import os
import unittest
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import stockstats_utils, interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


@pytest.mark.unit
class TestLoadOhlcvNoPoison(unittest.TestCase):
    def setUp(self):
        self._tmp = os.path.join(os.path.dirname(__file__), "_tmp_cache")
        os.makedirs(self._tmp, exist_ok=True)
        set_config({"data_cache_dir": self._tmp})

    def tearDown(self):
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)

    def test_empty_download_raises_and_does_not_cache(self):
        empty = pd.DataFrame()
        with mock.patch.object(stockstats_utils.yf, "download", return_value=empty) as dl:
            with self.assertRaises(NoMarketDataError):
                stockstats_utils.load_ohlcv("FAKE", "2026-01-01")
        # Nothing should have been written to the cache.
        self.assertEqual(os.listdir(self._tmp), [])

        # A second call must re-attempt the fetch (no poisoned cache served).
        with mock.patch.object(stockstats_utils.yf, "download", return_value=empty) as dl2:
            with self.assertRaises(NoMarketDataError):
                stockstats_utils.load_ohlcv("FAKE", "2026-01-01")
            self.assertTrue(dl2.called)


def _ohlcv_through(end_date: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", end_date)
    closes = [400.0 + i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(dates),
        }
    )


def _cache_path(cache_dir: str, symbol: str, today: str) -> str:
    today_dt = pd.Timestamp(today)
    start = (today_dt - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = today_dt.strftime("%Y-%m-%d")
    return os.path.join(cache_dir, f"{symbol}-YFin-data-{start}-{end}.csv")


@pytest.mark.unit
class TestLoadOhlcvStaleCache(unittest.TestCase):
    def setUp(self):
        self._tmp = os.path.join(os.path.dirname(__file__), "_tmp_stale_cache")
        os.makedirs(self._tmp, exist_ok=True)
        set_config({"data_cache_dir": self._tmp})

    def tearDown(self):
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)

    def test_stale_cache_triggers_refetch(self):
        stale = _ohlcv_through("2026-06-09")
        stale.to_csv(_cache_path(self._tmp, "MSFT", "2026-06-11"), index=False)
        fresh = _ohlcv_through("2026-06-10")

        with (
            mock.patch(
                "tradingagents.dataflows.stockstats_utils.pd.Timestamp.today",
                return_value=pd.Timestamp("2026-06-11"),
            ),
            mock.patch.object(stockstats_utils.yf, "download", return_value=fresh) as dl,
        ):
            result = stockstats_utils.load_ohlcv("MSFT", "2026-06-11")

        self.assertTrue(dl.called)
        self.assertEqual(result.iloc[-1]["Date"], pd.Timestamp("2026-06-10"))

    def test_fresh_cache_is_reused(self):
        fresh = _ohlcv_through("2026-06-10")
        fresh.to_csv(_cache_path(self._tmp, "MSFT", "2026-06-11"), index=False)

        with (
            mock.patch(
                "tradingagents.dataflows.stockstats_utils.pd.Timestamp.today",
                return_value=pd.Timestamp("2026-06-11"),
            ),
            mock.patch.object(stockstats_utils.yf, "download") as dl,
        ):
            result = stockstats_utils.load_ohlcv("MSFT", "2026-06-11")

        dl.assert_not_called()
        self.assertEqual(result.iloc[-1]["Date"], pd.Timestamp("2026-06-10"))

    def test_historical_curr_date_does_not_force_live_refresh(self):
        history = _ohlcv_through("2026-06-10")
        history.to_csv(_cache_path(self._tmp, "MSFT", "2026-06-11"), index=False)

        with (
            mock.patch(
                "tradingagents.dataflows.stockstats_utils.pd.Timestamp.today",
                return_value=pd.Timestamp("2026-06-11"),
            ),
            mock.patch.object(stockstats_utils.yf, "download") as dl,
        ):
            result = stockstats_utils.load_ohlcv("MSFT", "2026-06-05")

        dl.assert_not_called()
        self.assertEqual(result.iloc[-1]["Date"], pd.Timestamp("2026-06-05"))


@pytest.mark.unit
class TestLoadOhlcvCacheHelpers(unittest.TestCase):
    def test_expected_latest_cache_date_caps_at_curr_date(self):
        curr = pd.Timestamp("2026-01-15")
        today = pd.Timestamp("2026-06-11")
        expected = stockstats_utils._expected_latest_cache_date(curr, today)
        self.assertEqual(expected, pd.Timestamp("2026-01-15"))

    def test_is_cache_stale_when_row_missing(self):
        stale = _ohlcv_through("2026-06-09")
        self.assertTrue(
            stockstats_utils._is_cache_stale(
                stale,
                pd.Timestamp("2026-06-11"),
                pd.Timestamp("2026-06-11"),
            )
        )

    def test_max_ohlcv_date_ignores_nan_close_tail(self):
        good = _ohlcv_through("2026-06-12")
        nan_tail = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-06-15")],
                "Open": [float("nan")],
                "High": [float("nan")],
                "Low": [float("nan")],
                "Close": [float("nan")],
                "Volume": [45_098_048],
            }
        )
        cached = pd.concat([good, nan_tail], ignore_index=True)
        self.assertEqual(
            stockstats_utils._max_ohlcv_date(cached),
            pd.Timestamp("2026-06-12"),
        )
        self.assertTrue(
            stockstats_utils._is_cache_stale(
                cached,
                pd.Timestamp("2026-06-16"),
                pd.Timestamp("2026-06-16"),
            )
        )

    def test_format_missing_session_label_for_premarket_today(self):
        label = stockstats_utils.format_missing_session_label(
            "2026-06-16",
            curr_date="2026-06-16",
            latest_session=pd.Timestamp("2026-06-15"),
        )
        self.assertIn("Session not yet available", label)

    def test_format_missing_session_label_for_weekend(self):
        label = stockstats_utils.format_missing_session_label(
            "2026-06-14",
            curr_date="2026-06-16",
            latest_session=pd.Timestamp("2026-06-12"),
        )
        self.assertIn("weekend or holiday", label)


@pytest.mark.unit
class TestYFinanceStockData(unittest.TestCase):
    def test_get_yfin_data_online_uses_patched_ohlcv(self):
        from tradingagents.dataflows import y_finance

        ohlcv = _ohlcv_through("2026-06-15")
        with (
            mock.patch(
                "tradingagents.dataflows.y_finance.load_ohlcv",
                return_value=ohlcv,
            ),
            mock.patch(
                "tradingagents.dataflows.y_finance.pd.Timestamp.today",
                return_value=pd.Timestamp("2026-06-16"),
            ),
        ):
            out = y_finance.get_YFin_data_online("SPY", "2026-06-01", "2026-06-16")

        self.assertIn("Latest complete session available: 2026-06-15", out)
        self.assertIn("2026-06-16 has no complete daily bar yet", out)
        self.assertIn("2026-06-15", out)


@pytest.mark.unit
class TestLoadOhlcvNanTailCache(unittest.TestCase):
    def setUp(self):
        self._tmp = os.path.join(os.path.dirname(__file__), "_tmp_nan_tail_cache")
        os.makedirs(self._tmp, exist_ok=True)
        set_config({"data_cache_dir": self._tmp})

    def tearDown(self):
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)

    def test_nan_close_tail_triggers_refetch(self):
        good = _ohlcv_through("2026-06-12")
        nan_tail = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-06-15")],
                "Open": [float("nan")],
                "High": [float("nan")],
                "Low": [float("nan")],
                "Close": [float("nan")],
                "Volume": [45_098_048],
            }
        )
        pd.concat([good, nan_tail], ignore_index=True).to_csv(
            _cache_path(self._tmp, "SPY", "2026-06-16"), index=False
        )
        bulk = _ohlcv_through("2026-06-12")
        bulk_nan = pd.concat([bulk, nan_tail], ignore_index=True)
        tail_patch = _ohlcv_through("2026-06-15")
        tail_patch = tail_patch[tail_patch["Date"] >= pd.Timestamp("2026-06-15")]

        def fake_download(ticker, start, end, **kwargs):
            if start >= "2026-06-13":
                return tail_patch.set_index("Date")
            return bulk_nan.set_index("Date")

        with (
            mock.patch(
                "tradingagents.dataflows.stockstats_utils.pd.Timestamp.today",
                return_value=pd.Timestamp("2026-06-16"),
            ),
            mock.patch.object(stockstats_utils.yf, "download", side_effect=fake_download) as dl,
        ):
            result = stockstats_utils.load_ohlcv("SPY", "2026-06-16")

        self.assertGreaterEqual(dl.call_count, 2)
        self.assertEqual(result.iloc[-1]["Date"], pd.Timestamp("2026-06-15"))


@pytest.mark.unit
class TestOhlcvTailPatch(unittest.TestCase):
    def test_merge_prefers_valid_close_over_nan(self):
        base = _ohlcv_through("2026-06-12")
        nan_row = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-06-15")],
                "Open": [751.85],
                "High": [756.67],
                "Low": [751.76],
                "Close": [float("nan")],
                "Volume": [45_098_048],
            }
        )
        tail = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-06-15")],
                "Open": [751.85],
                "High": [756.67],
                "Low": [751.76],
                "Close": [754.83],
                "Volume": [45_098_048],
            }
        )
        merged = stockstats_utils._merge_ohlcv_frames(
            pd.concat([base, nan_row], ignore_index=True), tail
        )
        june15 = merged[merged["Date"] == pd.Timestamp("2026-06-15")].iloc[0]
        self.assertEqual(float(june15["Close"]), 754.83)

    def test_patch_incomplete_tail_fetches_missing_sessions(self):
        bulk = _ohlcv_through("2026-06-12")
        nan_row = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-06-15")],
                "Open": [751.85],
                "High": [756.67],
                "Low": [751.76],
                "Close": [float("nan")],
                "Volume": [45_098_048],
            }
        )
        tail = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-06-15")],
                "Open": [751.85],
                "High": [756.67],
                "Low": [751.76],
                "Close": [754.83],
                "Volume": [45_098_048],
            }
        )

        with mock.patch.object(
            stockstats_utils,
            "_download_yahoo_ohlcv",
            return_value=tail,
        ) as tail_dl:
            patched = stockstats_utils._patch_incomplete_tail(
                pd.concat([bulk, nan_row], ignore_index=True),
                "SPY",
                pd.Timestamp("2026-06-16"),
                pd.Timestamp("2026-06-16"),
            )

        tail_dl.assert_called_once()
        self.assertEqual(
            stockstats_utils._max_ohlcv_date(patched),
            pd.Timestamp("2026-06-15"),
        )


@pytest.mark.unit
class TestRouteToVendorSentinel(unittest.TestCase):
    def test_no_data_from_all_vendors_returns_sentinel(self):
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, "GC=F", "no rows")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_no_data}
        with mock.patch.dict(
            interface.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "XAUUSD+", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("XAUUSD+", result)
        self.assertIn("GC=F", result)
        self.assertIn("Do not estimate", result)

    def test_unconfigured_fallback_does_not_mask_no_data(self):
        # When the primary vendor reports no data and the fallback is simply
        # unavailable (e.g. missing API key -> raises), the no-data sentinel
        # must win rather than the fallback's incidental error crashing out.
        def raises_no_data(symbol, *a, **k):
            raise NoMarketDataError(symbol, symbol, "no rows")

        def raises_unavailable(symbol, *a, **k):
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

        patched = {"yfinance": raises_no_data, "alpha_vantage": raises_unavailable}
        with mock.patch.dict(
            interface.VENDOR_METHODS, {"get_stock_data": patched}, clear=False
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "FAKE", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)


if __name__ == "__main__":
    unittest.main()
