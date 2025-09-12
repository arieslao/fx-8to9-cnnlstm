import os, json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from utils import load_config, get_env_list
from data import concat_pairs
from features import add_indicators, feature_cols
from labels import make_8to9_label
from windows import build_sequences
import tensorflow as tf

def main():
    cfg = load_config()
    tickers = get_env_list("FX_TICKERS", [])
    if not tickers: raise ValueError("FX_TICKERS env required")

    raw = concat_pairs(
        tickers,
        cfg["data"]["test_start"],
        cfg["data"]["test_end"],
        interval=cfg["data"]["interval"],
        tz_name=cfg["data"]["timezone"]
    )
    feat = add_indicators(raw)
    labels = make_8to9_label(
        feat, cfg["label"]["target_window"]["start_hour"],
        cfg["label"]["target_window"]["end_hour"],
        cfg["label"]["direction_threshold"]
    )

    fcols = feature_cols()
    X, y, meta = build_sequences(feat, labels, fcols, seq_len=cfg["model"]["seq_len"])

    model = tf.keras.models.load_model(os.path.join(cfg["runtime"]["model_dir"], "cnn_lstm_fx.h5"))
    p = model.predict(X).ravel()
    yhat = (p >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yhat))
    }
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
