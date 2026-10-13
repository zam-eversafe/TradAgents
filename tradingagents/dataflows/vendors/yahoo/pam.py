"""Price Action Manipulation (PAM) setups from Yahoo OHLCV."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

PAM_REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class PamEvent:
    date: str
    setup: str
    bias: str
    key_level: float | None
    confirmation: str
    evidence: str


def analyze_price_action_manipulation(
    symbol: str,
    curr_date: str,
    look_back_days: int = 120,
    key_level_window: int = 20,
) -> str:
    """Analyze Price Action Manipulation (PAM) setups from cached OHLCV data."""
    data = load_ohlcv(symbol, curr_date)
    return analyze_price_action_manipulation_frame(
        data=data,
        symbol=symbol,
        curr_date=curr_date,
        look_back_days=look_back_days,
        key_level_window=key_level_window,
    )


def analyze_price_action_manipulation_frame(
    data: pd.DataFrame,
    symbol: str,
    curr_date: str,
    look_back_days: int = 120,
    key_level_window: int = 20,
) -> str:
    """Analyze PAM setups from a supplied OHLCV frame."""
    if look_back_days < 10:
        raise ValueError("look_back_days must be at least 10 for PAM analysis")
    if key_level_window < 3:
        raise ValueError("key_level_window must be at least 3 for PAM analysis")

    df = _prepare_pam_ohlcv(data, curr_date, look_back_days)
    if df.empty:
        return f"## Price Action Manipulation (PAM) for {symbol.upper()}\n\nNo OHLCV data available through {curr_date}."

    if len(df) <= key_level_window:
        return (
            f"## Price Action Manipulation (PAM) for {symbol.upper()}\n\n"
            f"Only {len(df)} rows are available. PAM needs more history than the "
            f"{key_level_window}-bar key-level window to identify traps and stop hunts."
        )

    events = _detect_pam_events(df, key_level_window)
    latest = df.iloc[-1]
    latest_prev = df.iloc[-2]
    latest_support, latest_resistance = _pam_key_levels(df, len(df) - 1, key_level_window)
    compression = _latest_pam_compression(df)
    recent_events = events[-8:]

    return _render_pam_report(
        symbol=symbol,
        curr_date=curr_date,
        look_back_days=look_back_days,
        key_level_window=key_level_window,
        latest=latest,
        latest_prev=latest_prev,
        latest_support=latest_support,
        latest_resistance=latest_resistance,
        compression=compression,
        events=recent_events,
        total_events=len(events),
    )


def _prepare_pam_ohlcv(data: pd.DataFrame, curr_date: str, look_back_days: int) -> pd.DataFrame:
    missing = [column for column in PAM_REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {missing}")

    curr_date_dt = pd.to_datetime(curr_date)
    before = curr_date_dt - pd.DateOffset(days=look_back_days)

    df = data.loc[:, PAM_REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0)
    df = df[(df["Date"] >= before) & (df["Date"] <= curr_date_dt)]
    df = df.sort_values("Date").reset_index(drop=True)

    df["Range"] = df["High"] - df["Low"]
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["UpperWick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["LowerWick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Return"] = df["Close"].pct_change()
    df["AvgVolume20"] = df["Volume"].rolling(20, min_periods=3).mean().shift(1)
    df["AvgRange20"] = df["Range"].rolling(20, min_periods=3).mean().shift(1)
    df["AvgAbsReturn20"] = df["Return"].abs().rolling(20, min_periods=3).mean().shift(1)
    df["Sma20"] = df["Close"].rolling(20, min_periods=3).mean()

    return df


def _detect_pam_events(df: pd.DataFrame, key_level_window: int) -> list[PamEvent]:
    events: list[PamEvent] = []
    start_idx = max(1, key_level_window)

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        support, resistance = _pam_key_levels(df, idx, key_level_window)
        if pd.isna(support) or pd.isna(resistance) or support >= resistance:
            continue

        confirmation = _pam_confirmation_label(row, prev, support, resistance)
        volume_spike = _is_pam_volume_spike(row)
        range_expansion = _is_pam_range_expansion(row)
        close = float(row["Close"])
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])

        bull_trap = high > resistance and close < resistance
        bear_trap = low < support and close > support
        long_upper_rejection = _is_pam_long_upper_rejection(row) and high > resistance
        long_lower_rejection = _is_pam_long_lower_rejection(row) and low < support

        if bull_trap:
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup="Bull Trap / Breakout Manipulation",
                    bias="Bearish reversal risk",
                    key_level=resistance,
                    confirmation=confirmation,
                    evidence=(
                        f"High pierced resistance at {_pam_money(resistance)} but closed back below it at {_pam_money(close)}"
                        f"{_pam_volume_text(volume_spike)}."
                    ),
                )
            )

        if bear_trap:
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup="Bear Trap / Breakdown Manipulation",
                    bias="Bullish reversal risk",
                    key_level=support,
                    confirmation=confirmation,
                    evidence=(
                        f"Low swept support at {_pam_money(support)} but closed back above it at {_pam_money(close)}"
                        f"{_pam_volume_text(volume_spike)}."
                    ),
                )
            )

        if long_upper_rejection:
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup="Stop Hunt / Liquidity Grab",
                    bias="Bearish",
                    key_level=resistance,
                    confirmation=confirmation,
                    evidence=(
                        f"Upper wick was {_pam_ratio(row['UpperWick'], row['Range'])} of the candle range after sweeping buy-side liquidity"
                        f"{_pam_volume_text(volume_spike)}."
                    ),
                )
            )

        if long_lower_rejection:
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup="Stop Hunt / Liquidity Grab",
                    bias="Bullish",
                    key_level=support,
                    confirmation=confirmation,
                    evidence=(
                        f"Lower wick was {_pam_ratio(row['LowerWick'], row['Range'])} of the candle range after sweeping sell-side liquidity"
                        f"{_pam_volume_text(volume_spike)}."
                    ),
                )
            )

        if _is_pam_pump_dump(df, idx, volume_spike):
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup="Pump & Dump",
                    bias="Bearish",
                    key_level=resistance,
                    confirmation=confirmation,
                    evidence="Abnormal markup on elevated volume was followed by rapid giveback within the next few bars.",
                )
            )

        if _is_pam_distribution_shakeout(df, idx, range_expansion):
            direction = "bullish" if close > open_price else "bearish"
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup="Distribution Shakeout",
                    bias=f"{direction.title()} expansion from range",
                    key_level=None,
                    confirmation=confirmation,
                    evidence="Recent whipsaw compression inside a range resolved with an expanded-range momentum candle.",
                )
            )

        continuation = _pam_continuation_setup(df, idx, support, resistance)
        if continuation:
            events.append(
                PamEvent(
                    date=_pam_date(row),
                    setup=continuation,
                    bias="Trend continuation",
                    key_level=resistance if continuation == "UC1" else support,
                    confirmation=confirmation,
                    evidence="Small shakeout near a key level resolved back in the direction of the prevailing trend.",
                )
            )

    return _dedupe_pam_events(events)


def _pam_key_levels(df: pd.DataFrame, idx: int, key_level_window: int) -> tuple[float, float]:
    prior = df.iloc[max(0, idx - key_level_window):idx]
    if prior.empty:
        return float("nan"), float("nan")
    return float(prior["Low"].min()), float(prior["High"].max())


def _pam_confirmation_label(row: pd.Series, prev: pd.Series, support: float, resistance: float) -> str:
    labels: list[str] = []

    if _is_pam_long_lower_rejection(row):
        labels.append("Pin Bar (hammer)")
    if _is_pam_long_upper_rejection(row):
        labels.append("Pin Bar (shooting star)")
    if _is_pam_bullish_engulfing(row, prev):
        labels.append("Mark Bar (bullish engulfing)")
    if _is_pam_bearish_engulfing(row, prev):
        labels.append("Mark Bar (bearish engulfing)")
    if _is_pam_inside_bar(row, prev):
        labels.append("Baby Bar inside Mother Bar")
    if _is_pam_ice_cream_bar(row, support, resistance):
        labels.append("Ice Cream Bar failed breakout")

    return ", ".join(labels) if labels else "No EXE confirmation"


def _is_pam_long_lower_rejection(row: pd.Series) -> bool:
    body = max(float(row["Body"]), 0.01)
    candle_range = max(float(row["Range"]), 0.01)
    return float(row["LowerWick"]) >= body * 2 and float(row["LowerWick"]) >= candle_range * 0.45


def _is_pam_long_upper_rejection(row: pd.Series) -> bool:
    body = max(float(row["Body"]), 0.01)
    candle_range = max(float(row["Range"]), 0.01)
    return float(row["UpperWick"]) >= body * 2 and float(row["UpperWick"]) >= candle_range * 0.45


def _is_pam_bullish_engulfing(row: pd.Series, prev: pd.Series) -> bool:
    return (
        row["Close"] > row["Open"]
        and prev["Close"] < prev["Open"]
        and row["Open"] <= prev["Close"]
        and row["Close"] >= prev["Open"]
    )


def _is_pam_bearish_engulfing(row: pd.Series, prev: pd.Series) -> bool:
    return (
        row["Close"] < row["Open"]
        and prev["Close"] > prev["Open"]
        and row["Open"] >= prev["Close"]
        and row["Close"] <= prev["Open"]
    )


def _is_pam_inside_bar(row: pd.Series, prev: pd.Series) -> bool:
    return row["High"] <= prev["High"] and row["Low"] >= prev["Low"]


def _is_pam_ice_cream_bar(row: pd.Series, support: float, resistance: float) -> bool:
    close = float(row["Close"])
    open_price = float(row["Open"])
    midpoint = (float(row["High"]) + float(row["Low"])) / 2
    failed_above = row["High"] > resistance and close < midpoint and close < open_price
    failed_below = row["Low"] < support and close > midpoint and close > open_price
    return failed_above or failed_below


def _is_pam_volume_spike(row: pd.Series) -> bool:
    avg_volume = row.get("AvgVolume20")
    return pd.notna(avg_volume) and float(avg_volume) > 0 and float(row["Volume"]) >= float(avg_volume) * 1.5


def _is_pam_range_expansion(row: pd.Series) -> bool:
    avg_range = row.get("AvgRange20")
    return pd.notna(avg_range) and float(avg_range) > 0 and float(row["Range"]) >= float(avg_range) * 1.4


def _is_pam_pump_dump(df: pd.DataFrame, idx: int, volume_spike: bool) -> bool:
    row = df.iloc[idx]
    avg_abs_return = row.get("AvgAbsReturn20")
    if pd.isna(avg_abs_return) or not volume_spike:
        return False

    move = float(row["Return"]) if pd.notna(row["Return"]) else 0.0
    if move < max(0.04, float(avg_abs_return) * 2.5):
        return False

    future = df.iloc[idx + 1:min(len(df), idx + 4)]
    if future.empty:
        return False

    return float(future["Close"].min()) <= float(row["Open"])


def _is_pam_distribution_shakeout(df: pd.DataFrame, idx: int, range_expansion: bool) -> bool:
    if idx < 8 or not range_expansion:
        return False

    window = df.iloc[idx - 8:idx]
    mean_close = float(window["Close"].mean())
    if mean_close <= 0:
        return False

    range_width = float(window["High"].max() - window["Low"].min()) / mean_close
    direction_changes = (window["Close"].diff().dropna().map(lambda value: 1 if value > 0 else -1).diff().abs() > 0).sum()
    return range_width <= 0.08 and direction_changes >= 3


def _pam_continuation_setup(df: pd.DataFrame, idx: int, support: float, resistance: float) -> str | None:
    row = df.iloc[idx]
    if pd.isna(row.get("Sma20")) or idx < 5:
        return None

    recent = df.iloc[idx - 4:idx]
    close = float(row["Close"])
    open_price = float(row["Open"])
    candle_range = max(float(row["Range"]), 0.01)
    body_share = float(row["Body"]) / candle_range

    uptrend = close > float(row["Sma20"]) and df["Sma20"].iloc[idx] > df["Sma20"].iloc[idx - 3]
    downtrend = close < float(row["Sma20"]) and df["Sma20"].iloc[idx] < df["Sma20"].iloc[idx - 3]
    recent_sell_side_sweep = recent["Low"].min() < support and close > resistance * 0.98
    recent_buy_side_sweep = recent["High"].max() > resistance and close < support * 1.02
    strong_body = body_share >= 0.65

    if uptrend and recent_sell_side_sweep and close > open_price and strong_body:
        return "UC1"
    if downtrend and recent_buy_side_sweep and close < open_price and strong_body:
        return "DC1"
    return None


def _latest_pam_compression(df: pd.DataFrame) -> str:
    if len(df) < 6:
        return "Insufficient data"

    recent = df.tail(6)
    mother = recent.iloc[0]
    babies = recent.iloc[1:]
    if not babies.empty and (babies["High"] <= mother["High"]).all() and (babies["Low"] >= mother["Low"]).all():
        return "Mother/Baby price compression active"

    recent_range = float(recent["High"].max() - recent["Low"].min())
    mean_close = float(recent["Close"].mean())
    if mean_close > 0 and recent_range / mean_close <= 0.04:
        return "Tight range compression"

    return "No active compression"


def _dedupe_pam_events(events: Iterable[PamEvent]) -> list[PamEvent]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[PamEvent] = []
    for event in events:
        key = (event.date, event.setup, event.bias)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _render_pam_report(
    symbol: str,
    curr_date: str,
    look_back_days: int,
    key_level_window: int,
    latest: pd.Series,
    latest_prev: pd.Series,
    latest_support: float,
    latest_resistance: float,
    compression: str,
    events: list[PamEvent],
    total_events: int,
) -> str:
    lines = [
        f"## Price Action Manipulation (PAM) Strategy for {symbol.upper()}",
        "",
        f"Analysis window: {look_back_days} calendar days through {curr_date}.",
        f"Key-level window: prior {key_level_window} trading bars.",
        "",
        "### Current Context",
        f"- Latest close: {_pam_money(float(latest['Close']))} on {_pam_date(latest)}.",
        f"- Nearest support: {_pam_money(latest_support)}.",
        f"- Nearest resistance: {_pam_money(latest_resistance)}.",
        f"- Compression state: {compression}.",
        f"- Latest candle confirmation: {_pam_confirmation_label(latest, latest_prev, latest_support, latest_resistance)}.",
        "",
        "### Recent PAM Events",
    ]

    if not events:
        lines.append("- No high-confidence PAM events detected in the current lookback window.")
    else:
        for event in events:
            level = f" near {_pam_money(event.key_level)}" if event.key_level is not None else ""
            lines.append(
                f"- {event.date}: **{event.setup}** ({event.bias}){level}. "
                f"Confirmation: {event.confirmation}. Evidence: {event.evidence}"
            )

    lines.extend(
        [
            "",
            "### Execution Notes",
            "- Treat PAM as context-first: traps matter most at support/resistance, weekly/monthly extremes, or compressed ranges.",
            "- Valid entries should wait for EXE confirmation such as pin bars, engulfing mark bars, or failed-breakout reversal candles.",
            "- Use volume spikes as supporting evidence of institutional activity, not as a standalone signal.",
            "- Risk planning should target at least a 2:1 reward/risk profile from the manipulation candle or reclaimed key level.",
            "",
            f"Total PAM events detected: {total_events}.",
        ]
    )

    return "\n".join(lines)


def _pam_date(row: pd.Series) -> str:
    value = row["Date"]
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _pam_money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def _pam_ratio(part: float, whole: float) -> str:
    if whole == 0 or pd.isna(whole):
        return "0%"
    return f"{float(part) / float(whole):.0%}"


def _pam_volume_text(volume_spike: bool) -> str:
    return " with a volume spike" if volume_spike else ""
