# src/train.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import load_config
from src.data_sheet import concat_pairs_sheet
from src.features import to_interval
from src.labels import make_window_label

# -----------------------
# Repro & small helpers
# -----------------------
def set_repro(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

# -----------------------
# Feature engineering
# -----------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, robust features on OHLCV:
      - pct change on Close
      - rolling z-scores for Close over 5 and 10 bars
      - high-low range and body size
      - rolling volatility (std of returns)
    Works on both 1h and 4h inputs.
    """
    if df.empty:
        return df

    feat = df.copy()
    feat["ret_close_1"] = feat.groupby("Ticker")["Close"].pct_change()

    # z-scores
    for w in (5, 10):
        roll_mean = feat.groupby("Ticker")["Close"].transform(lambda s: s.rolling(w, min_periods=3).mean())
        roll_std  = feat.groupby("Ticker")["Close"].transform(lambda s: s.rolling(w, min_periods=3).std())
        feat[f"z_close_{w}"] = (feat["Close"] - roll_mean) / (roll_std.replace(0, np.nan))

    # ranges/body
    feat["hl_range"] = (feat["High"] - feat["Low"]) / feat["Close"].replace(0, np.nan)
    feat["body"]     = (feat["Close"] - feat["Open"]) / feat["Close"].replace(0, np.nan)

    # rolling volatility of returns
    feat["vol_10"] = feat.groupby("Ticker")["ret_close_1"].transform(lambda s: s.rolling(10, min_periods=5).std())

    # Replace inf/NaN with 0 after feature creation (safe default)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat

# -----------------------
# Dataset builder
# -----------------------
def build_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) where each label row at timestamp t uses the *previous* seq_len bars
    strictly before t (no leakage). Operates per ticker.
    """
    X, y = [], []
    # Align by ticker, then for each label timestamp extract rolling window
    features = features.sort_index()
    labels   = labels.sort_index()
    for tkr, lab_tkr in labels.groupby("Ticker"):
        feat_tkr = features[features["Ticker"] == tkr]
        if feat_tkr.empty:
            continue
        idx = feat_tkr.index

        for ts, row in lab_tkr.iterrows():
            # end of window is strictly before the label's start timestamp
            end_loc = idx.searchsorted(ts)  # insertion point
            start_loc = end_loc - seq_len
            if start_loc < 0:
                continue
            window = feat_tkr.iloc[start_loc:end_loc]
            if len(window) != seq_len:
                continue
            X.append(window[feature_cols].to_numpy(dtype=np.float32))
            y.append(int(row["label"]))

    if not X:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.stack(X)
    y = np.array(y, dtype=np.int32)
    return X, y

# -----------------------
# Model
# -----------------------
def build_cnn_lstm(input_steps: int, input_feats: int, cfg: Dict) -> keras.Model:
    inp = keras.Input(shape=(input_steps, input_feats))
    x = layers.Conv1D(filters=cfg["model"]["cnn_filters"], kernel_size=cfg["model"]["cnn_kernel"], padding="causal", activation="relu")(inp)
    x = layers.Dropout(cfg["model"]["dropout"])(x)
    x = layers.LSTM(cfg["model"]["lstm_units"], return_sequences=False)(x)
    x = layers.Dropout(cfg["model"]["dropout"])(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["model"]["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )
    return model

# -----------------------
# Main train entry
# -----------------------
def main():
    set_repro(42)
    ensure_dirs()
    cfg = load_config()

    # ----------------- Load raw 1H data from Google Sheets -----------------
    pairs = (os.getenv("FX_TICKERS") or "EURUSD=X,GBPUSD=X").split(",")
    pairs = [p.strip() for p in pairs if p.strip()]

    df_all = concat_pairs_sheet(
        tickers=pairs,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["train_end"],
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df_all.empty:
        raise SystemExit("No rows loaded from sheet for the training window.")

    # ----------------- Resample to interval (4h) if requested --------------
    df_all = to_interval(df_all, cfg["data"]["interval"], cfg["data"]["timezone"])

    # ----------------- Feature engineering ---------------------------------
    feat_df = build_features(df_all)

    # Choose feature columns (exclude label-like columns)
    base_cols = ["Open","High","Low","Close","Volume"]
    extra_cols = [c for c in feat_df.columns if c not in base_cols + ["Ticker"]]
    feature_cols = base_cols + extra_cols

    # ----------------- Labels for (start_hour -> end_hour) ------------------
    labels = make_window_label(
        feat_df,
        cfg["label"]["target_window"]["start_hour"],   # e.g., 5
        cfg["label"]["target_window"]["end_hour"],     # e.g., 13
        cfg["label"]["direction_threshold"],           # e.g., 0.0
        ticker_col="Ticker",
        price_col="Close",
        trading_days_only=cfg["data"]["trading_days_only"],
    )
    if labels.empty:
        raise SystemExit("No labels produced for the requested window/hours. Check data coverage and hours.")

    # ----------------- Train vs Test split by date --------------------------
    tz = feat_df.index.tz
    train_start = pd.Timestamp(cfg["data"]["train_start"], tz=tz)
    train_end   = pd.Timestamp(cfg["data"]["train_end"],   tz=tz)
    test_start  = pd.Timestamp(cfg["data"]["test_start"],  tz=tz) if "test_start" in cfg["data"] else None
    test_end    = pd.Timestamp(cfg["data"]["test_end"],    tz=tz) if "test_end"   in cfg["data"] else None

    labels_train = labels[(labels.index >= train_start) & (labels.index <= train_end)]
    labels_test  = pd.DataFrame()
    if test_start is not None and test_end is not None:
        labels_test = labels[(labels.index >= test_start) & (labels.index <= test_end)]

    # ----------------- Build sequences -------------------------------------
    seq_len = int(cfg["model"]["seq_len"])
    X_train, y_train = build_sequences(feat_df, labels_train, seq_len, feature_cols)

    if len(X_train) == 0:
        raise SystemExit("No training sequences could be built. Try reducing seq_len or verifying data density.")

    # Optional: validation from tail of train set (also controlled by validation_split)
    print(f"[INFO] Train sequences: {X_train.shape}, positives={y_train.sum()}, negatives={(y_train==0).sum()}")

    # ----------------- Model & Training ------------------------------------
    model = build_cnn_lstm(seq_len, len(feature_cols), cfg)
    history = model.fit(
        X_train,
        y_train,
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        validation_split=cfg["train"]["validation_split"],
        shuffle=cfg["train"]["shuffle"],
        verbose=1,
    )

    # ----------------- Save model (keras + optional h5) ---------------------
    model.save("models/cnn_lstm_fx.keras")
    #try:
    #    model.save("models/cnn_lstm_fx.h5")
    #except Exception as e:
    #    print("[WARN] Could not save .h5 fallback:", repr(e))

    # ----------------- Persist training artifacts ---------------------------
    with open("artifacts/train_history.json", "w") as f:
        json.dump(history.history, f)
    with open("artifacts/train_shapes.json", "w") as f:
        json.dump({
            "X_train": list(X_train.shape),
            "y_train": int(y_train.shape[0]),
            "seq_len": seq_len,
            "n_features": len(feature_cols)
        }, f)
    with open("artifacts/feature_columns.txt", "w") as f:
        for c in feature_cols:
            f.write(f"{c}\n")

    print("[OK] Training complete. Saved models to models/ and logs to artifacts/.")

if __name__ == "__main__":
    main()
