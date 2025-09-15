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
        ws.append_row(["RunId","as_of_london","pair","snapped_start","snapped_end","desired_start","desired_end","prob_up","pred_label"])
    ws.append_rows(rows, value_input_option="USER_ENTERED")

def _now_london():
    return datetime.now(pytz.timezone("Europe/London"))

def _nearest_index_time(idx: pd.DatetimeIndex, target: pd.Timestamp, tol_hours: float = 2.1) -> datetime | None:
    pos = idx.searchsorted(target)
    best_ts, best_diff = None, None
    for j in (pos - 1, pos, pos + 1):
        if 0 <= j < len(idx):
            ts = idx[j]
            diff = abs(ts - target)
            if best_diff is None or diff < best_diff:
                best_ts, best_diff = ts, diff
    if best_ts is not None and best_diff <= pd.Timedelta(hours=tol_hours):
        return best_ts.to_pydatetime()
    return None

def main():
    cfg = load_config()
    os.makedirs("artifacts", exist_ok=True)

    # Load model (.keras preferred)
     MODEL_PATH = Path("models/cnn_lstm_fx.keras")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing: models/cnn_lstm_fx.keras")
    model = keras.saving.load_model(MODEL_PATH)

    tz = pytz.timezone(cfg["data"]["timezone"])
    as_of = _now_london().replace(minute=0, second=0, microsecond=0)

    start_hour = int(cfg["label"]["target_window"]["start_hour"])  # 9
    end_hour   = int(cfg["label"]["target_window"]["end_hour"])    # 13
    desired_start = as_of.replace(hour=start_hour)
    desired_end   = as_of.replace(hour=end_hour)
    # if running after end-hour, roll to next business day
    if as_of.hour >= end_hour:
        desired_start += timedelta(days=1)
        desired_end   += timedelta(days=1)

    seq_len = int(cfg["model"]["seq_len"])
    lookback_hours = int(cfg["data"]["lookback_hours"])
    hist_hours = lookback_hours + 48
    start_hist = (desired_start - timedelta(hours=hist_hours)).strftime("%Y-%m-%d")
    end_hist   = (desired_end + timedelta(hours=4)).strftime("%Y-%m-%d")

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

    df = to_interval(df, cfg["data"]["interval"], cfg["data"]["timezone"])

    # Features (same as train/eval)
    feat = df.copy()
    feat["ret_close_1"] = feat.groupby("Pair")["Close"].pct_change()
    for w in (5, 10):
        roll_mean = feat.groupby("Pair")["Close"].transform(lambda s: s.rolling(w, min_periods=3).mean())
        roll_std  = feat.groupby("Pair")["Close"].transform(lambda s: s.rolling(w, min_periods=3).std())
        feat[f"z_close_{w}"] = (feat["Close"] - roll_mean) / (roll_std.replace(0, np.nan))
    feat["hl_range"] = (feat["High"] - feat["Low"]) / feat["Close"].replace(0, np.nan)
    feat["body"]     = (feat["Close"] - feat["Open"]) / feat["Close"].replace(0, np.nan)
    feat["vol_10"]   = feat.groupby("Pair")["ret_close_1"].transform(lambda s: s.rolling(10, min_periods=5).std())
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    base_cols = ["Open","High","Low","Close","Volume"]
    extra     = [c for c in feat.columns if c not in base_cols + ["Pair"]]
    feature_cols = base_cols + extra

    rows_out = []
    for tkr, g in feat.groupby("Pair"):
        g = g.sort_index()
        idx = g.index

        snapped_start = _nearest_index_time(idx, pd.Timestamp(desired_start, tz=tz))
        snapped_end   = _nearest_index_time(idx, pd.Timestamp(desired_end, tz=tz))

        # Prefer the "next" bar after snapped_start if it's ~4h ahead
        if snapped_start is not None:
            pos = idx.searchsorted(pd.Timestamp(snapped_start, tz=tz))
            if pos + 1 < len(idx):
                candidate_end = idx[pos + 1].to_pydatetime()
                delta_h = (candidate_end - snapped_start).total_seconds() / 3600.0
                if 2.0 <= delta_h <= 6.0:
                    snapped_end = candidate_end

        if snapped_start is None or snapped_end is None or snapped_end <= snapped_start:
            continue

        # build latest sequence ending strictly before snapped_start (no leakage)
        end_loc = idx.searchsorted(pd.Timestamp(snapped_start, tz=tz))
        start_loc = end_loc - seq_len
        if start_loc < 0 or end_loc <= 0:
            continue
        window = g.iloc[start_loc:end_loc]
        if len(window) != seq_len:
            continue

        X = window[feature_cols].to_numpy(dtype=np.float32)[None, ...]
        prob_up = float(model.predict(X, verbose=0).reshape(-1)[0])
        pred = 1 if prob_up >= 0.5 else 0

        rows_out.append([
            os.getenv("GITHUB_RUN_ID", "local"),
            as_of.isoformat(),
            tkr,
            pd.Timestamp(snapped_start, tz=tz).isoformat(),
            pd.Timestamp(snapped_end,   tz=tz).isoformat(),
            pd.Timestamp(desired_start, tz=tz).isoformat(),
            pd.Timestamp(desired_end,   tz=tz).isoformat(),
            round(prob_up, 6),
            pred,
        ])

    if not rows_out:
        raise SystemExit("No pairs had sufficient history to score snapped 09→13 predictions.")

    _append_preds_rows(cfg["data"]["sheet"]["id"], "preds_8to9", rows_out)
    print(f"[OK] Wrote {len(rows_out)} predictions (snapped 09→13) to preds_8to9.")


if __name__ == "__main__":
    main()
