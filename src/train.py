# src/train.py
from __future__ import annotations

import os
import json
import numpy as np
import tensorflow as tf

from src.utils import load_config, ensure_dirs, get_env_list
from src.features import add_indicators, feature_cols
from src.labels import make_8to9_label
from src.windows import build_sequences
from src.model import build_cnn_lstm

# Data providers
from src.data import concat_pairs as concat_pairs_yf          # Yahoo fallback
from src.data_dk import concat_pairs_dk                       # Dukascopy primary


def _fetch(provider: str, tickers, start, end, interval, tz_name):
    """
    Unified fetch wrapper that returns a single DataFrame with columns:
    ['Open','High','Low','Close','Volume','Ticker'] indexed by tz-aware datetime.
    """
    provider = (provider or "dukascopy").lower()
    print(f"[INFO] Data provider: {provider}")
    if provider == "dukascopy":
        df = concat_pairs_dk(tickers, start, end, tz_name=tz_name)
    else:
        df = concat_pairs_yf(tickers, start, end, interval=interval, tz_name=tz_name)
    return df


def main():
    # ---- Load config & prepare dirs
    cfg = load_config()
    ensure_dirs(cfg["runtime"]["out_dir"], cfg["runtime"]["model_dir"])

    # ---- Read tickers from env (Repo → Settings → Actions → Variables → FX_TICKERS)
    tickers = get_env_list("FX_TICKERS", [])
    if not tickers:
        raise ValueError(
            "FX_TICKERS env required (e.g., EURUSD=X,GBPUSD=X,USDJPY=X). "
            "Set it in repo Settings → Secrets and variables → Actions → Variables."
        )
    print(f"[INFO] Training on {len(tickers)} pair(s): {tickers}")

    # ---- Fetch training data
    raw = _fetch(
        provider=cfg["data"].get("provider", "dukascopy"),
        tickers=tickers,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["train_end"],
        interval=cfg["data"]["interval"],
        tz_name=cfg["data"]["timezone"],
    )
    if raw.empty:
        raise RuntimeError("No data fetched for ANY ticker in train range. Check provider/date range.")

    print(f"[INFO] Fetched rows: {len(raw):,}. Timezone on index: {raw.index.tz}")
    print(f"[INFO] Raw columns: {list(raw.columns)}")

    # ---- Build features
    feat = add_indicators(raw)
    feat.index.name = "ts"  # harmless; labels.py no longer depends on it

    print(f"[INFO] Feature rows (post-indicator dropna): {len(feat):,}")

    # ---- Create 8→9 London labels (keep only the 08:00 samples with next-hour target)
    labels = make_8to9_label(
        feat,
        cfg["label"]["target_window"]["start_hour"],
        cfg["label"]["target_window"]["end_hour"],
        cfg["label"]["direction_threshold"],
    )
    print(f"[INFO] Label samples (08:00 rows with 09:00 targets): {len(labels):,}")

    # ---- Build sequences for CNN-LSTM
    fcols = feature_cols()
    seq_len = cfg["model"]["seq_len"]
    X, y, meta = build_sequences(feat, labels, fcols, seq_len=seq_len)
    if len(X) == 0:
        raise RuntimeError(
            "No sequences built for training. "
            "Causes: (a) insufficient history before 08:00, (b) strict dropna from indicators, "
            "(c) too-long seq_len vs available data."
        )

    n_features = len(fcols)
    print(f"[INFO] Training tensors — X: {X.shape}, y: {y.shape}, features: {n_features}, seq_len: {seq_len}")

    # ---- Build model
    model = build_cnn_lstm(
        seq_len=seq_len,
        n_features=n_features,
        cnn_filters=cfg["model"]["cnn_filters"],
        cnn_kernel=cfg["model"]["cnn_kernel"],
        lstm_units=cfg["model"]["lstm_units"],
        dropout=cfg["model"]["dropout"],
        lr=cfg["model"]["lr"],
    )

    # ---- Train
    history = model.fit(
        X, y,
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        validation_split=cfg["train"]["validation_split"],
        shuffle=cfg["train"]["shuffle"],
        verbose=2,
    )

    # ---- Save artifacts
    model_path = os.path.join(cfg["runtime"]["model_dir"], "cnn_lstm_fx.h5")
    model.save(model_path)
    meta_path = os.path.join(cfg["runtime"]["out_dir"], "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_samples": int(len(y)),
                "tickers": tickers,
                "seq_len": seq_len,
                "features": fcols,
                "provider": cfg["data"].get("provider", "dukascopy"),
                "train_start": cfg["data"]["train_start"],
                "train_end": cfg["data"]["train_end"],
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved model → {model_path}")
    print(f"[INFO] Saved meta  → {meta_path}")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    # Ensure unbuffered output in CI if needed: set PYTHONUNBUFFERED=1 in workflow env
    main()
