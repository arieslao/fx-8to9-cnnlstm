# src/labels.py
from __future__ import annotations
import pandas as pd

def make_window_label(
    df: pd.DataFrame,
    start_hour: int,
    end_hour: int,
    threshold: float = 0.0,
    *,
    ticker_col: str = "Ticker",
    price_col: str = "Close",
    trading_days_only: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Ticker","label","y","Close_start","Close_end","delta"])

    for c in (ticker_col, price_col):
        if c not in df.columns:
            raise ValueError(f"labels.make_window_label: missing column '{c}'")

    work = df[[ticker_col, price_col]].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col])

    h = work.index.hour
    dfa = work[h == start_hour].copy()
    dfb = work[h == end_hour].copy()

    if trading_days_only:
        dfa = dfa[dfa.index.dayofweek < 5]
        dfb = dfb[dfb.index.dayofweek < 5]

    dfa["Date"] = dfa.index.normalize()
    dfb["Date"] = dfb.index.normalize()

    left  = dfa.rename(columns={price_col: "Close_start"})
    right = dfb.rename(columns={price_col: "Close_end"})

    left_reset = left.reset_index()
    idx_name   = left.index.name or left_reset.columns[0]
    left_reset = left_reset.rename(columns={idx_name: "ts"})

    right_reset = right.reset_index()[[ticker_col, "Date", "Close_end"]]

    merged = (
        left_reset
        .merge(right_reset, on=[ticker_col, "Date"], how="inner", validate="m:1")
        .set_index("ts")
        .sort_index()
    )

    merged["delta"] = merged["Close_end"] - merged["Close_start"]
    merged["label"] = (merged["delta"] > float(threshold)).astype("int8")
    merged["y"]     = merged["label"].astype("int8")

    out = merged[[ticker_col, "label", "y", "Close_start", "Close_end", "delta"]].copy()
    out.index.name = None
    return out

# Back-compat alias (not used now, but harmless to keep)
def make_8to9_label(df, *args, **kwargs):
    return make_window_label(df, 8, 9, *args, **kwargs)
