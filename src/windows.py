from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


def build_sequences(
    feat: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    Create (X, y) by taking a rolling window of features that ends at the label timestamp.
    - feat: index=ts, columns contain 'Pair' and feature_cols
    - labels: columns ['ts','Pair','label'] (binary)
    """
    if "Pair" not in feat.columns:
        raise ValueError("feat must include 'Pair' column")
    if not {"ts", "Pair"}.issubset(labels.columns):
        raise ValueError("labels must include 'ts' and 'Pair'")
    lab_col = "label" if "label" in labels.columns else (
        "y" if "y" in labels.columns else None
    )
    if lab_col is None:
        raise ValueError("labels must include a 'label' column (or 'y').")

    # Ensure types
    feat = feat.sort_index()
    labels = labels.copy()
    labels["ts"] = pd.to_datetime(labels["ts"])

    X, y, meta = [], [], []

    # Group features by pair for quick slicing
    for pair, grp in feat.groupby("Pair"):
        grp = grp.sort_index()
        # convenience array for speed
        for _, row in labels.loc[labels["Pair"] == pair].iterrows():
            ts = row["ts"]
            end = ts  # the label timestamp aligns to the last row of the window
            start = ts - pd.Timedelta(hours=seq_len)

            window = grp.loc[(grp.index > start) & (grp.index <= end)]
            if len(window) < seq_len:
                # not enough history, skip
                continue

            X.append(window[feature_cols].tail(seq_len).to_numpy(dtype=float))
            y.append(int(row[lab_col]))
            meta.append((pair, ts))

    if not X:
        raise RuntimeError("After alignment, no training windows were created. Check date ranges and feature columns.")

    return np.stack(X), np.array(y, dtype=int), meta
