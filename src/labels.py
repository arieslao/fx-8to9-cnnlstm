# src/labels.py
from __future__ import annotations
from typing import Optional
import pandas as pd

def make_8to9_label(
    df: pd.DataFrame,
    start_hour: int = 8,          # now allowed positionally
    end_hour: int = 9,            # now allowed positionally
    threshold: float = 0.0,       # now allowed positionally
    *,
    ticker_col: str = "Ticker",   # keyword-only below
    price_col: str = "Close",
    trading_days_only: bool = True,
) -> pd.DataFrame:
    """
    Build labels for the 08:00→09:00 London window.

    Input
      df: DataFrame with DateTimeIndex (tz-aware), columns include [ticker_col, price_col]
    Output
      DataFrame indexed by the 08:00 timestamps with columns:
        ['Ticker', 'y', 'Close_08', 'Close_09', 'delta']
      where y = 1 if (Close_09 - Close_08) > threshold else 0
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "y", "Close_08", "Close_09", "delta"])

    # sanity
    for c in (ticker_col, price_col):
        if c not in df.columns:
            raise ValueError(f"labels.make_8to9_label: missing column '{c}'")

    work = df[[ticker_col, price_col]].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col])

    h = work.index.hour
    df8 = work[h == start_hour].copy()
    df9 = work[h == end_hour].copy()

    if trading_days_only:
        df8 = df8[df8.index.dayofweek < 5]
        df9 = df9[df9.index.dayofweek < 5]

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

    merged["delta"] = merged["Close_09"] - merged["Close_08"]
    merged["y"] = (merged["delta"] > float(threshold)).astype("int8")

    out = merged[[ticker_col, "y", "Close_08", "Close_09", "delta"]].copy()
    out.index.name = None
    return out
