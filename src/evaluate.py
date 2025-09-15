# src/evaluate.py
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import keras

from src.utils import load_config
from src.data_sheet import concat_pairs_sheet
from src.features import to_interval
from src.labels import make_window_label_auto

# --- Sheets writer (inline) ---
import gspread
from google.oauth2.service_account import Credentials

def _sheets_client():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def _append_eval_rows(sheet_id: str, worksheet: str, run_id: str, window_start: str, window_end: str, metrics: dict):
    gc = _sheets_client()
    ws = gc.open_by_key(sheet_id).worksheet(worksheet)
    if ws.acell("A1").value is None:
        ws.append_row(["RunId","window_start","window_end","metric","value","created_at"])
    now = datetime.now(timezone.utc).isoformat()
    rows = [[run_id, window_start, window_end, k, float(v), now] for k, v in metrics.items()]
    ws.append_rows(rows, value_input_option="USER_ENTERED")

def _safe_div(a, b):
    return (a / b) if b else 0.0

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) else 0.0
    denom = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def build_sequences(features: pd.DataFrame, labels: pd.DataFrame, seq_len: int, feature_cols: list[str]):
    X, y = [], []
    features = features.sort_index()
    labels = labels.sort_index()
    for tkr, lab_tkr in labels.groupby("Paier"):
        feat_tkr = features[features["Pair"] == tkr]
        if feat_tkr.empty: continue
        idx = feat_tkr.index
        for ts, row in lab_tkr.iterrows():
            end_loc = idx.searchsorted(ts)
            start_loc = end_loc - seq_len
            if start_loc < 0: continue
            window = feat_tkr.iloc[start_loc:end_loc]
            if len(window) != seq_len: continue
            X.append(window[feature_cols].to_numpy(dtype=np.float32))
            y.append(int(row["label"]))
    if not X:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(X), np.array(y, dtype=np.int32)

def main():
    os.makedirs("artifacts", exist_ok=True)
    cfg = load_config()

    # load model
    MODEL_PATH = Path("models/cnn_lstm_fx.keras")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing: models/cnn_lstm_fx.keras")
    model = keras.saving.load_model(MODEL_PATH)

    pairs = (os.getenv("FX_TICKERS") or "EURUSD=X,GBPUSD=X").split(",")
    pairs = [p.strip() for p in pairs if p.strip()]

    df = concat_pairs_sheet(
        tickers=pairs,
        start=cfg["data"]["test_start"],
        end=cfg["data"]["test_end"],
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df.empty:
        raise SystemExit("No test rows from sheet.")

    df = to_interval(df, cfg["data"]["interval"], cfg["data"]["timezone"])

    # features consistent with training
    feat = df.copy()
    feat["ret_close_1"] = feat.groupby("Pair")["Close"].pct_change()
    for w in (5, 10):
        roll_mean = feat.groupby("Paier")["Close"].transform(lambda s: s.rolling(w, min_periods=3).mean())
        roll_std  = feat.groupby("Pair")["Close"].transform(lambda s: s.rolling(w, min_periods=3).std())
        feat[f"z_close_{w}"] = (feat["Close"] - roll_mean) / (roll_std.replace(0, np.nan))
    feat["hl_range"] = (feat["High"] - feat["Low"]) / feat["Close"].replace(0, np.nan)
    feat["body"]     = (feat["Close"] - feat["Open"]) / feat["Close"].replace(0, np.nan)
    feat["vol_10"]   = feat.groupby("Pair")["ret_close_1"].transform(lambda s: s.rolling(10, min_periods=5).std())
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    base_cols = ["Open","High","Low","Close","Volume"]
    extra_cols = [c for c in feat.columns if c not in base_cols + ["Pair"]]
    feature_cols = base_cols + extra_cols

    # --- auto-snap labels for 09->13 ---
    labels = make_window_label_auto(
        feat,
        cfg["label"]["target_window"]["start_hour"],
        cfg["label"]["target_window"]["end_hour"],
        ticker_col="Ticker",
        price_col="Close",
        trading_days_only=cfg["data"]["trading_days_only"],
        duration_hours=4,
        snap_tolerance_hours=2.1,
    )
    if labels.empty:
        raise SystemExit("No labels found in test window after snapping.")

    seq_len = int(cfg["model"]["seq_len"])
    X, y_true = build_sequences(feat, labels, seq_len, feature_cols)
    if len(X) == 0:
        raise SystemExit("No sequences could be built for evaluation.")

    y_prob = model.predict(X, verbose=0).reshape(-1)
    metrics = compute_metrics(y_true, y_prob, thr=0.5)

    print("===== Evaluation Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    with open("artifacts/eval_metrics.json","w") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k,v in metrics.items()}, f, indent=2)

    run_id = os.getenv("GITHUB_RUN_ID", datetime.now(timezone.utc).strftime("local-%Y%m%d%H%M%S"))
    _append_eval_rows(
        cfg["data"]["sheet"]["id"],
        "eval_daily",
        run_id,
        cfg["data"]["test_start"],
        cfg["data"]["test_end"],
        metrics
    )
    print("[OK] Evaluation written to artifacts/ and Sheets tab 'eval_daily'.")


if __name__ == "__main__":
    main()
