from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from src.data_sheet import concat_pairs_sheet, append_rows
from src.labels import make_9to13_label
from src.windows import build_sequences

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

def main():
    cfg = load_config()
    pairs = pairs_from_env() or ["EURUSD=X"]
    model_path = Path("models/cnn_lstm_fx.keras")
    if not model_path.exists():
        raise FileNotFoundError("Model missing: models/cnn_lstm_fx.keras")

    model = keras.saving.load_model(model_path)

    df = concat_pairs_sheet(
        pairs=pairs,
        start=cfg["data"]["test_start"],
        end=cfg["data"]["test_end"],
        tz_name=cfg["data"]["timezone"],
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet=cfg["data"]["sheet"]["worksheet"],
    )
    if df.empty:
        raise SystemExit("No eval rows from sheet.")

    df_feat = make_features(df)
    y_all = make_9to13_label(df_feat)
    df_lab = df_feat.dropna(subset=["label"]).copy()
    df_lab["label"] = df_lab["label"].astype(int)

    seq_len = int(cfg["model"]["seq_len"])
    feature_cols = ["Open","High","Low","Close","Volume","ret","hl_range"]

    X, t_end, pairs_list = build_sequences(df_feat, seq_len=seq_len, feature_cols=feature_cols)
    meta = pd.DataFrame({"t_end": t_end, "Pair": pairs_list}).set_index(["t_end","Pair"])
    targets = df_lab.set_index(df_lab.index.rename("t_end")).set_index("Pair", append=True)["label"]

    idx = meta.index.intersection(targets.index)
    if len(idx) == 0:
        raise SystemExit("No aligned windows for evaluation.")
    keep = meta.index.get_indexer(idx)
    X_eval = X[keep]
    y_true = targets.loc[idx].to_numpy(dtype="int32")

    proba = model.predict(X_eval, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype("int32")

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    print({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc})

    # Write a compact summary to eval_daily
    rows = [[
        pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        ",".join(pairs),
        acc, prec, rec, f1, mcc, "eval"
    ]]
    append_rows(
        sheet_id=cfg["data"]["sheet"]["id"],
        worksheet="eval_daily",
        header=["UTC Logged", "Pairs", "accuracy", "precision", "recall", "f1", "mcc", "run_id"],
        rows=rows,
    )

if __name__ == "__main__":
    main()
