from __future__ import annotations

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from keras import layers, models, optimizers
from src.data_sheet import concat_pairs_sheet
from src.labels import make_9to13_label
from src.windows import build_sequences

def load_config() -> dict:
    import yaml
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def pairs_from_env() -> list[str]:
    raw = os.getenv("FX_PAIRS") or os.getenv("FX_TICKERS") or ""
    return [p.strip() for p in raw.split(",") if p.strip()]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic returns and ranges; you can expand later
    out = df.copy()
    out["ret"] = out["Close"].pct_change()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Open"]
    out = out.dropna()
    return out

def main():
    cfg = load_config()
    Path("models").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    pairs = pairs_from_env() or ["EURUSD=X","GBPUSD=X","USDJPY=X"]
    print("[INFO] Training on pairs:", pairs)

    df = concat_pairs_sheet(
        pairs=pairs,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["train_end"],
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df.empty:
        raise SystemExit("No training rows from sheet. Check Sheet ID/tab, share, and date range.")

    df_feat = make_features(df)
    # Build labels only on 09:00 bars; other rows are NaN
    y_all = make_9to13_label(df_feat)
    df_feat["label"] = y_all

    # Keep only rows that have a label (i.e., 09:00 bars)
    df_lab = df_feat.dropna(subset=["label"]).copy()
    df_lab["label"] = df_lab["label"].astype(int)

    seq_len = int(cfg["model"]["seq_len"])
    feature_cols = ["Open","High","Low","Close","Volume","ret","hl_range"]

    # Build windows over *all* 4H rows, but only keep windows whose END is a labeled row
    X, t_end, pairs_list = build_sequences(df_feat, seq_len=seq_len, feature_cols=feature_cols)
    if len(t_end) == 0:
        raise SystemExit("No windows built. Check seq_len and data volume.")
    meta = pd.DataFrame({"t_end": t_end, "Pair": pairs_list})
    meta = meta.set_index(["t_end","Pair"])
    targets = df_lab.set_index(df_lab.index.rename("t_end")).set_index("Pair", append=True)["label"]

    # align X with y via index
    idx = meta.index.intersection(targets.index)
    keep = meta.index.get_indexer(idx)
    X = X[keep]
    y = targets.loc[idx].to_numpy(dtype="int32")

    if X.shape[0] == 0:
        raise SystemExit("After aligning windows with labels, nothing to train on.")

    # Build CNN-LSTM
    inputs = layers.Input(shape=(seq_len, len(feature_cols)))
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(cfg["model"]["lr"])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X, y,
        batch_size=int(cfg["train"]["batch_size"]),
        epochs=int(cfg["train"]["epochs"]),
        validation_split=float(cfg["train"]["validation_split"]),
        shuffle=True,
        verbose=2,
    )

    out_path = Path("models") / "cnn_lstm_fx.keras"
    model.save(out_path)
    print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    main()
