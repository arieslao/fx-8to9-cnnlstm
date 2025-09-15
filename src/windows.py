from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd

def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str]
) -> Tuple[np.ndarray, list[pd.Timestamp], list[str]]:
    """
    Build rolling sequences per Pair.
    Returns:
      X: [N, seq_len, F]
      t_end: list of timestamps corresponding to each window end
      pairs: list of Pair (length N)
    """
    X_list: List[np.ndarray] = []
    t_end: List[pd.Timestamp] = []
    pairs: List[str] = []

    for pair, g in df.groupby("Pair", sort=False):
        g = g.sort_index()
        arr = g[feature_cols].to_numpy(dtype="float32")
        # build windows
        for i in range(len(g) - seq_len + 1):
            X_list.append(arr[i : i + seq_len])
            t_end.append(g.index[i + seq_len - 1])
            pairs.append(pair)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, seq_len, len(feature_cols)), dtype="float32")
    return X, t_end, pairs
