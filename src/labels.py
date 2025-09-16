from __future__ import annotations

import pandas as pd
import numpy as np


def _infer_step_hours(idx: pd.DatetimeIndex) -> int:
    """Infer the bar step in hours (1h, 4h, etc.)."""
    if len(idx) < 3:
        return 1
    diffs = (idx[1:] - idx[:-1]).to_series().dt.total_seconds().dropna()
    # Use the most common difference
    step_sec = int(diffs.mode().iloc[0])
    return max(1, step_sec // 3600)


def make_8to9_label(
    df: pd.DataFrame,
    start_hour: int = 8,
    end_hour: int = 9,
) -> pd.DataFrame:
    """
    Given a price frame indexed by ts (tz-aware) and with a 'Pair' and 'Close' column,
    produce labels at rows where ts.hour == start_hour, comparing Close at start_hour vs end_hour.

    Returns columns: ['ts','Pair','label'] where label in {0,1}.
    """
    if "Pair" not in df.columns or "Close" not in df.columns:
        raise ValueError("Expected columns 'Pair' and 'Close' to build labels.")

    # Ensure ts is a real column (copy from index)
    df = df.sort_index()
    work = df[["Pair", "Close"]].copy()
    work["ts"] = work.index

    step_h = _infer_step_hours(work["ts"])
    delta_h = (end_hour - start_hour) % 24

    # We create a table of source rows at start_hour
    src = work[work["ts"].dt.hour == start_hour].copy()

    # Create a shifted view of Close for the target hour:
    # align target close back onto the source timestamp by subtracting the delta
    tgt = work.copy()
    tgt["ts"] = tgt["ts"] - pd.to_timedelta(delta_h, unit="h")
    tgt = tgt.rename(columns={"Close": "Close_t"})

    merged = src.merge(
        tgt[["ts", "Pair", "Close_t"]],
        on=["ts", "Pair"],
        how="left",
        validate="one_to_one",
    )

    # Build label: up = 1 if future close > source close, else 0
    merged["label"] = (merged["Close_t"] - merged["Close"] > 0).astype(int)

    # Keep only rows where we actually have a future close
    merged = merged.dropna(subset=["Close_t"])

    out = merged[["ts", "Pair", "label"]].copy()
    # Guarantee types
    out["ts"] = pd.to_datetime(out["ts"])
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)
