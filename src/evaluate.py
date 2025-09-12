# src/evaluate.py
import os
import json
import numpy as np
import tensorflow as tf

from src.utils import load_config, get_env_list
from src.features import add_indicators, feature_cols
from src.labels import make_8to9_label
from src.windows import build_sequences

# Providers
from src.data_sheet import concat_pairs_sheet
from src.data import concat_pairs as concat_pairs_yf
from src.data_dk import concat_pairs_dk


def _fetch(provider: str, tickers, start, end, interval, tz_name, cfg):
    provider = (provider or "sheet").lower()
    print(f"[INFO] Data provider: {provider}")
    if provider == "sheet":
        sheet_id = cfg["data"]["sheet"]["id"]
        worksheet = cfg["data"]["sheet"]["worksheet"]
        return concat_pairs_sheet(tickers, start, end, tz_name, sheet_id, worksheet)
    elif provider == "dukascopy":
        from src.data_dk import concat_pairs_dk
        return concat_pairs_dk(tickers, start, end, tz_name=tz_name)
    else:  # yahoo fallback
        from src.data import concat_pairs as concat_pairs_yf
        return concat_pairs_yf(tickers, start, end, interval=interval, tz_name=tz_name)



def _binary_metrics(y_true: np.ndarray, y_hat: np.ndarray):
    """
    Compute accuracy, precision, recall, F1, MCC without sklearn.
    y_true, y_hat: arrays of 0/1.
    """
    y_true = y_true.astype(int).ravel()
    y_hat = y_hat.astype(int).ravel()

    tp = int(((y_true == 1) & (y_hat == 1)).sum())
    tn = int(((y_true == 0) & (y_hat == 0)).sum())
    fp = int(((y_true == 0) & (y_hat == 1)).sum())
    fn = int(((y_true == 1) & (y_hat == 0)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def main():
    cfg = load_config()
    tickers = get_env_list("FX_TICKERS", [])
    if not tickers:
        raise ValueError("FX_TICKERS env required (e.g., EURUSD=X,GBPUSD=X,USDJPY=X)")

    raw = _fetch(
        provider=cfg["data"].get("provider", "sheet"),
        tickers=tickers,
        start=cfg["data"]["train_start"],
        end=cfg["data"]["train_end"],
        interval=cfg["data"]["interval"],
        tz_name=cfg["data"]["timezone"],
        cfg=cfg,
    )

    if raw.empty:
        raise RuntimeError("No data fetched for ANY ticker in test range. Check provider/date range.")

    feat = add_indicators(raw)
    labels = make_8to9_label(
        feat,
        cfg["label"]["target_window"]["start_hour"],
        cfg["label"]["target_window"]["end_hour"],
        cfg["label"]["direction_threshold"],
    )

    fcols = feature_cols()
    X, y, meta = build_sequences(
        feat, labels, fcols, seq_len=cfg["model"]["seq_len"]
    )
    if len(X) == 0:
        raise RuntimeError("No sequences built for evaluation (insufficient 08:00 samples).")

    model_path = os.path.join(cfg["runtime"]["model_dir"], "cnn_lstm_fx.h5")
    model = tf.keras.models.load_model(model_path)

    probs = model.predict(X, verbose=0).ravel()
    yhat = (probs >= 0.5).astype(int)

    metrics = _binary_metrics(y, yhat)

    # Print JSON for CI logs; also include a tiny preview of pairs and sample count
    out = {
        "n_samples": int(len(y)),
        "tickers": tickers,
        "metrics": metrics
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
