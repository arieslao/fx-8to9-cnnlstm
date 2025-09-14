# src/features.py
from __future__ import annotations
import pandas as pd

def to_interval(df: pd.DataFrame, interval: str, tz_name: str = "Europe/London") -> pd.DataFrame:
    """
    Convert 1H OHLCV per-pair data to the requested interval.

    Supported:
      - '1h' : passthrough
      - '4h' : resample to 4-hour bars anchored at 01:00 (London), yielding bars
               at 01,05,09,13,17,21 so that 05:00 and 13:00 exist.

    Expects:
      - tz-aware DateTimeIndex
      - columns: ['Open','High','Low','Close','Volume','Ticker']

    Returns a DataFrame with the same columns, resampled and sorted by index.
    """
    if df.empty or (interval and interval.lower() == "1h"):
        return df

    if interval.lower() != "4h":
        raise ValueError(f"Unsupported interval: {interval}")

    out = []
    # Resample per ticker to keep proper OHLCV semantics
    for t, d in df.groupby("Ticker"):
        d = d.sort_index()
        r = (
            d[["Open", "High", "Low", "Close", "Volume"]]
            .resample("4h", origin="start_day", offset="1h")
            .agg({
                "Open": "first",
                "High": "max",
                "Low":  "min",
                "Close":"last",
                "Volume":"sum"
            })
            .dropna(how="any")
        )
        r["Ticker"] = t
        out.append(r)

    return pd.concat(out).sort_index()
