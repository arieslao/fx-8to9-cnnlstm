# src/labels.py
from __future__ import annotations

import pandas as pd
from typing import Optional

"""
make_8to9_label
---------------
Given hourly FX candles indexed by a tz-aware DateTimeIndex (London time)
and a column 'Ticker', produce a binary label for each 08:00 bar that
indicates the direction from 08:00 -> 09:00 (Close_09 - Close_08).

Returns a DataFrame with:
  index  : the 08:00 timestamps
  columns: ['Ticker', 'y', 'Close_08', 'Close_09', 'delta']

where:
  y = 1 if (Close_09 - Close_08) > threshold else 0
"""

def make_8to9_label(
    df: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    price_col: str = "Close",
    start_hour: int = 8,
    end_hour: int = 9,
    threshold: float = 0.0,
    trading_days_only: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [ticker_col, price_col] and have a tz-aware DateTimeIndex
        aligned to Europe/London (08 and 09 mean local London times).
    ticker_col : str
    price_col : str
    start_hour : int
        08 by default.
    end_hour : int
        09 by default (one hour after start_hour).
    threshold : float
        Label is 1 if (Close_09 - Close_08) > threshold else 0.
    trading_days_only : bool
        If True, drop Saturdays/Sundays.

    Returns
    -------
    pd.DataFrame with index at the 08:00 timestamps and columns:
        ['Ticker', 'y', 'Close_08', 'Close_09', 'delta']
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "y", "Close_08", "Close_09", "delta"])

    # Defensive: make sure required cols exist and are numeric
    req = {ticker_col, price_col}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"labels.make_8to9_label: missing columns {missing} in df")

    work = df.copy()
    # Only keep the columns we need
    work = work[[ticker_col, price_col]].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col])

    # Filter to 08:00 and 09:00 rows
    h = work.index.hour
    df8 = work[h == start_hour].copy()
    df9 = work[h == end_hour].copy()

    if trading_days_only:
        # Monday=0 ... Sunday=6; drop Sat/Sun
        df8 = df8[(df8.index.dayofweek < 5)]
        df9 = df9[(df9.index.dayofweek < 5)]

    # Merge by (date, ticker)
    df8["Date"] = df8.index.normalize()
    df9["Date"] = df9.index.normalize()

    left = df8.rename(columns={price_col: "Close_08"})
    right = df9.rename(columns={price_col: "Close_09"})

    merged = (
        left.reset_index(drop=False)
            .merge(
                right[[ticker_col, "Date", "Close_09"]],
                on=[ticker_col, "Date"],
                how="inner",
                validate="m:1",
            )
            .set_index("index")
            .sort_index()
    )

    # Compute delta and label
    merged["delta"] = merged["Close_09"] - merged["Close_08"]
    merged["y"] = (merged["delta"] > float(threshold)).astype("int8")

    # Keep tidy columns
    out = merged[[ticker_col, "y", "Close_08", "Close_09", "delta"]]
    out.index.name = None
    return out
