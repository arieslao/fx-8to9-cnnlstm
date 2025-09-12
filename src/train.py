import os, json
import numpy as np
import pandas as pd
from src.utils import load_config, ensure_dirs, get_env_list
# OLD:
# from src.data import concat_pairs
# NEW:
from src.data import concat_pairs as concat_pairs_yf
from src.data_dk import concat_pairs_dk

from src.features import add_indicators, feature_cols
from src.labels import make_8to9_label
from src.windows import build_sequences
from src.model import build_cnn_lstm


def main():
    cfg = load_config()
    provider = cfg["data"].get("provider", "dukascopy").lower()
    
    if provider == "dukascopy":
        raw = concat_pairs_dk(
            tickers,
            cfg["data"]["train_start"],
            cfg["data"]["train_end"],
            tz_name=cfg["data"]["timezone"]
        )
    else:
        raw = concat_pairs_yf(
            tickers,
            cfg["data"]["train_start"],
            cfg["data"]["train_end"],
            interval=cfg["data"]["interval"],
            tz_name=cfg["data"]["timezone"]
        )
    
    if raw.empty:
        raise RuntimeError("No data fetched for ANY ticker. Check provider/date range.")


    feat = add_indicators(raw)
    label_df = make_8to9_label(
        feat, cfg["label"]["target_window"]["start_hour"],
        cfg["label"]["target_window"]["end_hour"],
        cfg["label"]["direction_threshold"]
    )

    fcols = feature_cols()
    seq_len = cfg["model"]["seq_len"]
    X, y, meta = build_sequences(feat, label_df, fcols, seq_len=seq_len)

    n_features = len(fcols)
    model = build_cnn_lstm(
        seq_len=seq_len,
        n_features=n_features,
        cnn_filters=cfg["model"]["cnn_filters"],
        cnn_kernel=cfg["model"]["cnn_kernel"],
        lstm_units=cfg["model"]["lstm_units"],
        dropout=cfg["model"]["dropout"],
        lr=cfg["model"]["lr"]
    )

    history = model.fit(
        X, y,
        batch_size=cfg["train"]["batch_size"],
        epochs=cfg["train"]["epochs"],
        validation_split=cfg["train"]["validation_split"],
        shuffle=cfg["train"]["shuffle"],
        verbose=2
    )

    # Save
    model_path = os.path.join(cfg["runtime"]["model_dir"], "cnn_lstm_fx.h5")
    model.save(model_path)

    with open(os.path.join(cfg["runtime"]["out_dir"], "train_meta.json"), "w") as f:
        json.dump({
            "n_samples": int(len(y)),
            "tickers": tickers,
            "seq_len": seq_len,
            "features": fcols
        }, f, indent=2)

    print(f"Saved model: {model_path}; samples: {len(y)}")

if __name__ == "__main__":
    main()
