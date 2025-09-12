# src/labels.py
from __future__ import annotations
import pandas as pd

def make_8to9_label(
    feat: pd.DataFrame,
    start_hour: int = 8,
    end_hour: int = 9,           # kept for signature symmetry; we use next-hour close
    direction_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build labels for the 08:00→09:00 London window.
    Returns a DataFrame indexed by timestamp ('ts') with columns: ['y','Ticker'].

    y = 1 if Close@09:00 - Close@08:00 > direction_threshold else 0
    """
    df = feat.copy()

    # Ensure tz-aware London index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/London")
    df = df.sort_index()

    # Next-hour close per pair
    df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)

    # Keep only 08:00 rows (the label is about the move into the next hour)
    m8 = df.index.hour == int(start_hour)
    lab = df.loc[m8, ["Close", "next_close", "Ticker"]].dropna().copy()

    # Direction label
    lab["y"] = (lab["next_close"] - lab["Close"] > float(direction_threshold)).astype("int8")

    # Shape for downstream: index named 'ts', columns ['y','Ticker']
    lab = lab.drop(columns=["Close", "next_close"])
    lab.index.name = "ts"
    return lab[["y", "Ticker"]]
