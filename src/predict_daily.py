# src/predict_daily.py
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz
import keras

from src.utils import load_config
from src.data_sheet import concat_pairs_sheet
from src.features import to_interval

# --- Sheets client (inline) ---
import gspread
from google.oauth2.service_account import Credentials

def _sheets_client():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def _append_preds_rows(sheet_id: str, worksheet: str, rows: list[list]):
    gc = _sheets_client()
    ws = gc.open_by_key(sheet_id).worksheet(worksheet)
    if ws.acell("A1").value is None:
        ws.append_row(["RunId","as_of_london","pair","target_start","target_end","prob_up","pred_label"])
    ws.append_rows(rows, value_input_option="USER_ENTERED")

def _now_london():
    return datetime.now(pytz.timezone("Europe/London"))

def main():
    cfg = load_config()
    os.makedirs("artifacts", exist_ok=True)

    # 1) Load model
    model_path = Path("models/cnn_lstm_fx.keras")
    if not model_path.exists():
        alt = Path("models/cnn_lstm_fx.h5")
        if alt.exists(): model_path = alt
        else: raise FileNotFoundError("Model missing: models/cnn_lstm_fx.keras (or .h5)")
    model = keras.saving.load_model(model_path)

    # 2) Timing
    tz = pytz.timezone(cfg["data"]["timezone"])
    now_lon = _now_london()
    as_of = now_lon.replace(minute=0, second=0, microsecond=0)

    # We intend to predict the window 09:00 -> 13:00 today
    start_hour = int(cfg["label"]["target_window"]["start_hour"])  # 9
    end_hour   = int(cfg["label"]["target_window"]["end_hour"])    # 13
    start_ts = as_of.replace(hour=start_hour)
    end_ts   = as_of.replace(hour=end_hour)
    if as_of.hour >= end_hour:   # if run after the window today, roll to next business day
        start_ts += timedelta(days=1)
        end_ts   += timedelta(days=1)

    # 3) Pull enough history (lookback + a buffer day)
    seq_len = int(cfg["model"]["seq_len"])
    lookback_hours = int(cfg["data"]["lookback_hours"])
    hist_hours = lookback_hours + 48
    start_hist = (start_ts - timedelta(hours=hist_hours)).strftime("%Y-%m-%d")
    end_hist   = (end_ts + timedelta(hours=4)).strftime("%Y-%m-%d")

    pairs = (os.getenv("FX_TICKERS") or "EURUSD=X,GBPUSD=X,USDJPY=X").split(",")
    pairs = [p.strip() for p in pairs if p.strip()]

    df = concat_pairs_sheet(
        tickers=pairs,
        start=start_hist,
        end=end_hist,
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df.empty:
        raise SystemExit("No rows from sheet for prediction window.")

    # 4) Interval passthrough + features (same transforms as training)
    df = to_interval(df, cfg["data"]["interval"], cfg["data"]["timezone"])

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

    base_cols = ["Open","High","Low","Close","Volume"]
    extra     = [c for c in feat.columns if c not in base_cols + ["Ticker"]]
    feature_cols = base_cols + extra

    # 5) Build latest sequence per pair ending strictly before start_ts (no leakage)
    rows_out = []
    for tkr, f in feat.groupby("Ticker"):
        f = f.sort_index()
        idx = f.index
        end_loc = idx.searchsorted(start_ts)  # first index >= start_ts
        start_loc = end_loc - seq_len
        if start_loc < 0 or end_loc <= 0:  # not enough history
            continue
        window = f.iloc[start_loc:end_loc]
        if len(window) != seq_len:
            continue
        X = window[feature_cols].to_numpy(dtype=np.float32)[None, ...]
        prob_up = float(model.predict(X, verbose=0).reshape(-1)[0])
        pred    = 1 if prob_up >= 0.5 else 0
        rows_out.append([
            os.getenv("GITHUB_RUN_ID", "local"),
            as_of.isoformat(),
            tkr,
            start_ts.isoformat(),
            end_ts.isoformat(),
            round(prob_up, 6),
            pred,
        ])

    if not rows_out:
        raise SystemExit("No pairs had sufficient history to score a 09→13 prediction.")

    # 6) Write to Sheet tab 'preds_8to9' (kept name for continuity)
    _append_preds_rows(
        cfg["data"]["sheet"]["id"],
        "preds_8to9",
        rows_out
    )
    print(f"[OK] Wrote {len(rows_out)} predictions to preds_8to9.")

if __name__ == "__main__":
    main()
