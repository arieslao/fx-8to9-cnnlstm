# src/predict_daily.py
import os
import json
import pandas as pd
import tensorflow as tf

# --- Loading keras ----
from pathlib import Path
import keras

model_path = Path("models/cnn_lstm_fx.keras")
if not model_path.exists():
    raise FileNotFoundError("Model missing: models/cnn_lstm_fx.keras")
model = keras.saving.load_model(model_path)
# --- continue loading others ---

from src.utils import load_config, now_london, get_env_list
from src.features import add_indicators, feature_cols

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



def should_run_now():
    """
    Allow execution only around 07:55–08:05 London so we publish the 8→9 prediction
    right at 08:00. Use --force to bypass this for testing.
    """
    now = now_london()
    return (now.hour == 7 and now.minute >= 55) or (now.hour == 8 and now.minute <= 5)


def main(force: bool = False):
    cfg = load_config()
    tickers = get_env_list("FX_TICKERS", [])
    if not tickers:
        raise ValueError("FX_TICKERS env required (e.g., EURUSD=X,GBPUSD=X,USDJPY=X)")

    if not force and not should_run_now():
        print("Skipping: outside 07:55–08:05 London window. Use --force to override.")
        return

    # Fetch just enough history to build the sequence ending at today's 08:00 London
    seq_len = cfg["model"]["seq_len"]
    end_london = now_london()
    start_london = end_london - pd.Timedelta(hours=seq_len + 6)  # buffer

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
        print("No fresh data fetched.")
        return

    feat = add_indicators(raw)
    fcols = feature_cols()

    model_path = os.path.join(cfg["runtime"]["model_dir"], "cnn_lstm_fx.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    model = tf.keras.models.load_model(model_path)

    # Build one sequence per ticker ending exactly at today's 08:00 London
    preds = []
    today = end_london.date()
    ts_0800 = pd.Timestamp(
        year=end_london.year, month=end_london.month, day=end_london.day,
        hour=8, minute=0, tz=feat.index.tz
    )

    for t in tickers:
        hist = feat[(feat["Ticker"] == t) & (feat.index <= ts_0800)].tail(seq_len)
        if len(hist) < seq_len:
            preds.append({"Ticker": t, "status": "insufficient_history"})
            continue

        X = hist[fcols].values.reshape(1, seq_len, len(fcols))
        prob_up = float(model.predict(X, verbose=0)[0][0])
        preds.append({
            "Ticker": t,
            "prob_up_8to9": round(prob_up, 4),
            "signal": "LONG" if prob_up >= 0.5 else "SHORT"
        })

    out_dir = cfg["runtime"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Dated file
    dated_path = os.path.join(out_dir, f"pred_{today}.json")
    with open(dated_path, "w") as f:
        json.dump({"date": str(today), "predictions": preds}, f, indent=2)

    # Rolling latest (for Streamlit)
    latest_path = os.path.join(out_dir, "pred_latest.json")
    with open(latest_path, "w") as f:
        json.dump({"date": str(today), "predictions": preds}, f, indent=2)

    print(json.dumps({"saved": dated_path, "saved_latest": latest_path, "predictions": preds}, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Run regardless of current time window")
    args = p.parse_args()
    main(force=args.force)
