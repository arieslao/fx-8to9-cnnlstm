from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np
import pytz
import keras
from src.data_sheet import concat_pairs_sheet, append_rows

def load_config() -> dict:
    import yaml
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def pairs_from_env() -> list[str]:
    raw = os.getenv("FX_PAIRS") or os.getenv("FX_TICKERS") or ""
    return [p.strip() for p in raw.split(",") if p.strip()]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["Close"].pct_change()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Open"]
    return out.dropna()

def anchor_next_0913_window(now_london: pd.Timestamp) -> pd.Timestamp:
    """Return the timestamp of the *start* of the next 09:00 4H candle."""
    base = now_london.floor("H")
    # 4h anchors in London: 01,05,09,13,17,21
    anchors = [1,5,9,13,17,21]
    h = base.hour
    next_h = next((a for a in anchors if a > h), anchors[0])
    ts = base.replace(hour=next_h, minute=0, second=0, microsecond=0)
    if next_h <= h:
        ts = ts + pd.Timedelta(days=1)
    return ts.replace(tzinfo=now_london.tz)

def main():
    cfg = load_config()
    pairs = pairs_from_env() or ["EURUSD=X"]
    model = keras.saving.load_model(Path("models/cnn_lstm_fx.keras"))

    # pull recent history (use test_start to "today")
    df = concat_pairs_sheet(
        pairs=pairs,
        start=cfg["data"]["test_start"],
        end=cfg["data"]["test_end"],
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df.empty:
        raise SystemExit("No rows to predict on.")

    df_feat = make_features(df)

    seq_len = int(cfg["model"]["seq_len"])
    feature_cols = ["Open","High","Low","Close","Volume","ret","hl_range"]

    # choose the point-in-time near 08:55 London
    london = pytz.timezone(cfg["data"]["timezone"])
    now_london = pd.Timestamp.utcnow().tz_convert(london) if pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(london)
    target_start = anchor_next_0913_window(now_london)

    # build the most recent window per Pair that ends *before* target_start
    rows_out = []
    for pair, g in df_feat.groupby("Pair"):
        g = g[g.index < target_start].sort_index()
        if len(g) < seq_len: continue
        X = g[feature_cols].iloc[-seq_len:].to_numpy(dtype="float32")[None, ...]
        p_up = float(model.predict(X, verbose=0).ravel()[0])
        direction = int(p_up >= 0.5)
        rows_out.append([
            target_start.strftime("%Y-%m-%d %H:%M:%S"),
            pair,
            direction,
            p_up,
            "predict_9to13"
        ])

    if rows_out:
        append_rows(
            sheet_id=cfg["data"]["sheet"]["id"],
            worksheet="preds_8to9",
            header=["Target start (London)","Pair","y_pred","proba_up","run_id"],
            rows=rows_out,
        )
        print(f"[OK] Wrote {len(rows_out)} predictions to preds_8to9")
    else:
        print("[WARN] No predictions written (insufficient history or no pairs).")

if __name__ == "__main__":
    main()
