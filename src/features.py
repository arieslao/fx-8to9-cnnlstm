# src/features.py
from __future__ import annotations
import pandas as pd

def to_interval(df: pd.DataFrame, interval: str, tz_name: str = "Europe/London") -> pd.DataFrame:
    """
    Data is already 4h in the Sheet. For '4h' we return as-is.
    Kept for future-proofing (e.g., if you later ingest 1h and resample).
    """
    if df.empty or (interval and interval.lower() in ("4h", "4hr", "4hour")):
        return df
    if interval.lower() == "1h":
        return df  # also passthrough if you ever switch tabs to 1h
    raise ValueError(f"Unsupported interval: {interval}")
