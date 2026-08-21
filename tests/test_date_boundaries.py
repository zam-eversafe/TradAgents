"""yfinance treats ``end`` as exclusive; we must request extra day(s) so the
requested end_date (and the current day) is actually included.

Regressions for #986 (current-day OHLCV excluded) and #987 (requested end_date
row omitted). Market-data extraction uses the patched ``load_ohlcv`` path so
incomplete Yahoo tails cannot strand the agent on an older session.
"""
import pandas as pd
import pytest

import tradingagents.dataflows.stockstats_utils as su
import tradingagents.dataflows.y_finance as yfin
from tradingagents.dataflows.config import set_config


@pytest.mark.unit
def test_get_yfin_uses_patched_ohlcv_and_keeps_requested_end(monkeypatch):
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-05-08", "2025-05-09"]),
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [1, 2],
        }
    )
    monkeypatch.setattr(yfin, "load_ohlcv", lambda symbol, curr_date: frame)
    out = yfin.get_YFin_data_online("AAPL", "2025-05-01", "2025-05-09")

    # Header still reflects the requested range, not an internal exclusive bound.
    assert "to 2025-05-09" in out
    assert "2025-05-09" in out


@pytest.mark.unit
def test_load_ohlcv_requests_inclusive_end(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    captured = {}

    def fake_download(symbol, start, end, **kwargs):
        captured["end"] = end
        idx = pd.to_datetime([pd.Timestamp.today().normalize()])
        return pd.DataFrame(
            {"Open": [100.0], "High": [100.0], "Low": [100.0],
             "Close": [100.0], "Volume": [1]},
            index=idx,
        )

    monkeypatch.setattr(su.yf, "download", fake_download)
    today = pd.Timestamp.today()
    su.load_ohlcv("AAPL", today.strftime("%Y-%m-%d"))

    expected_end = su._yahoo_download_end_date(today)
    assert captured["end"] == expected_end  # exclusive end includes today's row (#986)
