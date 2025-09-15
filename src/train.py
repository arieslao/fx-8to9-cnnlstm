# src/train.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import load_config
from src.data_sheet import concat_pairs_sheet
from src.features import to_interval
from src.labels import make_window_label

def set_repro(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    feat = df.copy()
    feat["ret_close_1"] = feat.groupby("Ticker")["Close"].pct_change()
    for w in (5, 10):
        roll_mean = feat.groupby("Ticker")["Close"].transform(lambda s: s.rolling(w, min_periods=3).mean())
        roll_std  = feat.groupby("Ticker")["Close"].transform(lambda s: s.rolling(w, min_periods=3).std())
        feat[f"z_close_{w}"] = (feat["Close"] - roll_mean) / (roll_std.replace(0, np.nan))
    feat["hl_range"] = (feat["High"] - feat["Low"]) / feat["Close"].replace(0, np.nan)
    feat["body"]     = (feat["Close"] - feat["Open"]) / feat["Close"].replace(0, np.nan)
    feat["vol_10"]   = feat.groupby("Ticker")["ret_close_1"].transform(lambda s: s.rolling(10, min_periods=5).std())
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat

def build_sequences(features: pd.DataFrame, labels: pd.DataFrame, seq_len: int, cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    features, labels = features.sort_index(), labels.sort_index()
    for tkr, lab_tkr in labels.groupby("Ticker"):
        f = features[features["Ticker"] == tkr]
        if f.empty: continue
        idx = f.index
        for ts, row in lab_tkr.iterrows():
            end_loc = idx.searchsorted(ts)  # strictly before label time
            start = end_loc - seq_len
            if start < 0: continue
            win = f.iloc[start:end_loc]
            if len(win) != seq_len: continue
            X.append(win[cols].to_numpy(dtype=np.float32))
            y.append(int(row["label"]))
    if not X:
        return np.empty((0, seq_len, len(cols)), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(X), np.array(y, dtype=np.int32)

def build_cnn_lstm(steps: int, feats: int, cfg: Dict) -> keras.Model:
    inp = keras.Input(shape=(steps, feats))
    x = layers.Conv1D(cfg["model"]["cnn_filters"], cfg["model"]["cnn_kernel"], padding="causal", activation="relu")(inp)
    x = layers.Dropout(cfg["model"]["dropout"])(x)
    x = layers.LSTM(cfg["model"]["lstm_units"])(x)
    x = layers.Dropout(cfg["model"]["dropout"])(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg["model"]["lr"]),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    return model

def main():
    set_repro(42); ensure_dirs()
    cfg = load_config()

    pairs = (os.getenv("FX_TICKERS") or "EURUSD=X,GBPUSD=X,USDJPY=X").split(",")
    pairs = [p.strip() for p in pairs if p.strip()]

    df = concat_pairs_sheet(
        tickers=pairs,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["train_end"],
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df.empty: raise SystemExit("No rows from sheet for training window.")

    # passthrough (still here for future-proofing)
    df = to_interval(df, cfg["data"]["interval"], cfg["data"]["timezone"])

    feat = build_features(df)
    base = ["Open","High","Low","Close","Volume"]
    extra = [c for c in feat.columns if c not in base + ["Ticker"]]
    feature_cols = base + extra

    labels = make_window_label(
        feat,
        cfg["label"]["target_window"]["start_hour"],  # 9
        cfg["label"]["target_window"]["end_hour"],    # 13
        cfg["label"]["direction_threshold"],
        ticker_col="Ticker",
        price_col="Close",
        trading_days_only=cfg["data"]["trading_days_only"],
    )
    if labels.empty:
        raise SystemExit("No labels for 09→13. Check that your 4h grid contains 09:00 and 13:00 closes.")

    seq_len = int(cfg["model"]["seq_len"])
    X, y = build_sequences(feat, labels, seq_len, feature_cols)
    if len(X) == 0:
        raise SystemExit("No train sequences. Consider reducing seq_len or checking date coverage.")

    model = build_cnn_lstm(seq_len, len(feature_cols), cfg)
    hist = model.fit(
        X, y,
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        validation_split=cfg["train"]["validation_split"],
        shuffle=cfg["train"]["shuffle"],
        verbose=1,
    )

    model.save("models/cnn_lstm_fx.keras")   # modern keras format
    with open("artifacts/train_history.json","w") as f: json.dump(hist.history, f)
    with open("artifacts/feature_columns.txt","w") as f: f.write("\n".join(feature_cols))
    print("[OK] Trained & saved to models/cnn_lstm_fx.keras")

if __name__ == "__main__":
    main()
