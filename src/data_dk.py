# src/data_dk.py
from __future__ import annotations
import pandas as pd
import numpy as np

# Try to import dukascopy-python; fail with a helpful message if missing
try:
    import dukascopy_python as dk  # pip install dukascopy-python
except Exception as e:
    dk = None
    _dk_import_error = e

def _ensure_dk():
    if dk is None:
        raise ImportError(
            "dukascopy-python is not available. Install it in requirements.txt "
            "(e.g., dukascopy-python==4.0.1). Original import error: "
            f"{_dk_import_error!r}"
        )

def _normalize(df: pd.DataFrame, ticker: str, tz_name: str) -> pd.DataFrame:
    # Expect columns: open, high, low, close, volume, time/index
    # Rename to our canonical schema and add Ticker
    cols = {c.lower(): c for c in df.columns}
    rename = {
        cols.get("open", "open"): "Open",
        cols.get("high", "high"): "High",
        cols.get("low", "low"): "Low",
        cols.get("close", "close"): "Close",
        cols.get("volume", "volume"): "Volume",
    }
    df = df.rename(columns=rename)
    # Index to tz-aware DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        # dukascopy-python typically returns a 'time' column if not indexed
        time_col = "time" if "time" in df.columns else None
        if time_col:
            df = df.set_index(pd.to_datetime(df[time_col], utc=True))
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    if tz_name:
        df.index = df.index.tz_convert(tz_name)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker
    return df.sort_index()

def fetch_pair_dk(ticker: str, start: str, end: str | None, tz_name: str) -> pd.DataFrame:
    """
    Fetch 1h candles for a single pair from Dukascopy and return normalized DF.
    """
    _ensure_dk()

    # dukascopy-python API wrapper: request hourly candles
    # If your code previously used minute/granularity enums, adapt here:
    # The library exposes instrument symbols and a get_price_history-style function.
    # We request hourly (60m) bars between start and end.
    start_ts = pd.Timestamp(start, tz=tz_name).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz=tz_name).tz_convert("UTC") if end else pd.Timestamp.utcnow().tz_localize("UTC")

    # Some packages expect ticker without "=X". Remove suffix if present.
    inst = ticker.replace("=X", "")

    # Hourly aggregation via dukascopy-python
    # NOTE: If your installed version exposes a different function name,
    # the smoke step/workflow check will make that obvious—adjust here to match.
    df = dk.get_price_history(
        instrument=inst,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        timeframe="1H"  # 1-hour bars
    )

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","Ticker"])

    return _normalize(df, ticker, tz_name)

def concat_pairs_dk(tickers: list[str], start: str, end: str | None, tz_name: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            df = fetch_pair_dk(t, start, end, tz_name)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[WARN] Dukascopy fetch failed for {t}: {e!r}")
    if not frames:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","Ticker"])
    return pd.concat(frames).sort_index()
