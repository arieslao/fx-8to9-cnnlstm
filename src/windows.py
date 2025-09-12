import numpy as np
import pandas as pd

def build_sequences(full_df: pd.DataFrame, label_df: pd.DataFrame, feature_cols, seq_len=72):
    """
    For each 08:00 sample (label_df index = 08:00 ts), grab previous `seq_len` hours
    from full_df for the same ticker, and build X,y.
    """
    X, y, meta = [], [], []

    # Reindex label_df to ensure datetime index
    for ts, row in label_df.iterrows():
        ticker = row["Ticker"]
        start = ts - pd.Timedelta(hours=seq_len)
        hist = full_df[(full_df["Ticker"]==ticker) & (full_df.index > start) & (full_df.index <= ts)]
        if len(hist) < seq_len:
            continue
        seq = hist[feature_cols].values[-seq_len:]
        X.append(seq)
        y.append(row["label"])
        meta.append({"ts": ts, "Ticker": ticker})

    return np.array(X), np.array(y), meta
