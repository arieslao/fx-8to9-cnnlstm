from __future__ import annotations
import pandas as pd

def make_8to9_label(
    df: pd.DataFrame,
    start_hour: int = 8,
    end_hour: int = 9,
    threshold: float = 0.0,
    *,
    ticker_col: str = "Ticker",
    price_col: str = "Close",
    trading_days_only: bool = True,
) -> pd.DataFrame:
    """
    Build labels for the 08:00→09:00 London window.
    Returns a DataFrame indexed by the 08:00 timestamp with columns:
      ['Ticker', 'label', 'y', 'Close_08', 'Close_09', 'delta']
    where label/y = 1 if (Close_09 - Close_08) > threshold else 0
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "label", "y", "Close_08", "Close_09", "delta"])

    # sanity
    for c in (ticker_col, price_col):
        if c not in df.columns:
            raise ValueError(f"labels.make_8to9_label: missing column '{c}'")

    work = df[[ticker_col, price_col]].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col])

    # 08:00 and 09:00 slices
    h = work.index.hour
    df8 = work[h == start_hour].copy()
    df9 = work[h == end_hour].copy()

    if trading_days_only:
        df8 = df8[df8.index.dayofweek < 5]
        df9 = df9[df9.index.dayofweek < 5]

    # Join on (Date, Ticker); keep the 08:00 timestamp as the index
    df8["Date"] = df8.index.normalize()
    df9["Date"] = df9.index.normalize()
    left = df8.rename(columns={price_col: "Close_08"})
    right = df9.rename(columns={price_col: "Close_09"})

    # reset indexes safely and rename the 08:00 index to 'ts'
    left_reset = left.reset_index()
    idx8_name = left.index.name or left_reset.columns[0]  # whatever pandas chose
    left_reset = left_reset.rename(columns={idx8_name: "ts"})

    right_reset = right.reset_index()[[ticker_col, "Date", "Close_09"]]

    merged = (
        left_reset
        .merge(right_reset, on=[ticker_col, "Date"], how="inner", validate="m:1")
        .set_index("ts")
        .sort_index()
    )

    merged["delta"] = merged["Close_09"] - merged["Close_08"]
    # primary column name expected by the rest of the pipeline
    merged["label"] = (merged["delta"] > float(threshold)).astype("int8")
    # keep 'y' as an alias for compatibility with any old call sites
    merged["y"] = merged["label"].astype("int8")

    out = merged[[ticker_col, "label", "y", "Close_08", "Close_09", "delta"]].copy()
    out.index.name = None
    return out
