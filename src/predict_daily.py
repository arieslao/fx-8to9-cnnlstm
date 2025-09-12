import os, json
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import load_config, now_london, get_env_list
from data import concat_pairs
from features import add_indicators, feature_cols
from windows import build_sequences

def should_run_now():
    # Run only around 07:55–08:05 London so we can publish the 8→9 prediction right at 08:00
    now = now_london()
    return (now.hour == 7 and now.minute >= 55) or (now.hour == 8 and now.minute <= 5)

def main(force=False):
    cfg = load_config()
    tickers = get_env_list("FX_TICKERS", [])
    if not tickers:
        raise ValueError("FX_TICKERS env required")

    if not force and not should_run_now():
        print("Skipping: outside 07:55–08:05 London window.")
        return

    # Fetch last (seq_len + 5) hours up to 'now' to ensure we capture the 08:00 bar
    seq_len = cfg["model"]["seq_len"]
    end = now_london()
    start = (end - pd.Timedelta(hours=seq_len + 6)).strftime("%Y-%m-%d %H:%M:%S")

    raw = concat_pairs(
        tickers,
        start,
        None,
        interval=cfg["data"]["interval"],
        tz_name=cfg["data"]["timezone"]
    )
    if raw.empty:
        print("No fresh data.")
        return

    feat = add_indicators(raw)
    fcols = feature_cols()
    model = tf.keras.models.load_model(os.path.join(cfg["runtime"]["model_dir"], "cnn_lstm_fx.h5"))

    # Build one sequence per ticker ending exactly at 08:00 today
    preds = []
    today = now_london().date()
    ts_0800 = pd.Timestamp(
        year=end.year, month=end.month, day=end.day, hour=8, minute=0, tz=feat.index.tz
    )

    for t in tickers:
        hist = feat[(feat["Ticker"]==t) & (feat.index <= ts_0800)].tail(seq_len)
        if len(hist) < seq_len:
            preds.append({"Ticker": t, "status": "insufficient_history"})
            continue
        X = hist[fcols].values.reshape(1, seq_len, len(fcols))
        prob_up = float(model.predict(X, verbose=0)[0][0])
        preds.append({"Ticker": t, "prob_up_8to9": round(prob_up,4), "signal": "LONG" if prob_up>=0.5 else "SHORT"})

    out_path = os.path.join(cfg["runtime"]["out_dir"], f"pred_{today}.json")
    os.makedirs(cfg["runtime"]["out_dir"], exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"date": str(today), "predictions": preds}, f, indent=2)

    print(json.dumps({"saved": out_path, "predictions": preds}, indent=2))

if __name__ == "__main__":
    # set force=True if you want to test outside the time window
    main(force=False)
