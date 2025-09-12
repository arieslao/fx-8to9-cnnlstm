# app.py
import os, glob, json
import pandas as pd
import streamlit as st

# --- Your exact pairs (deduped) ---
PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X",
    "EURJPY=X","USDCAD=X","USDCHF=X","NZDUSD=X",
]

st.set_page_config(page_title="FX 8→9 London Predictions", layout="centered")
st.title("FX 8→9 London Predictions")
st.caption("Shows the most recent predictions saved by the GitHub Action / predict script.")

# Prefer fixed filename if you added 'pred_latest.json'; otherwise use newest dated file
latest_fixed = os.path.join("artifacts", "pred_latest.json")
files = sorted(glob.glob(os.path.join("artifacts", "pred_*.json")))

data = None
src = None
if os.path.exists(latest_fixed):
    with open(latest_fixed, "r") as f:
        data = json.load(f)
    src = latest_fixed
elif files:
    with open(files[-1], "r") as f:
        data = json.load(f)
    src = files[-1]

if not data:
    st.warning("No prediction files found in artifacts/. Run the **Predict 8→9 London** workflow (or run locally) to generate one.")
    st.stop()

st.caption(f"Source: `{src}`")
df = pd.DataFrame(data.get("predictions", []))

# Filter to your pairs & tidy
df = df[df["Ticker"].isin(PAIRS)].copy()
if df.empty:
    st.info("Prediction file exists, but none of your configured pairs were found. Check that FX_TICKERS matches the pairs above.")
else:
    # Order by your PAIRS list
    df["order"] = df["Ticker"].apply(lambda t: PAIRS.index(t) if t in PAIRS else 999)
    df = df.sort_values("order").drop(columns=["order"])

    # Nice column order if present
    cols = [c for c in ["Ticker","prob_up_8to9","signal","status"] if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]

    # Display
    st.dataframe(df, use_container_width=True)

    if "prob_up_8to9" in df.columns:
        st.subheader("Probability Up 8→9 (per pair)")
        st.bar_chart(df.set_index("Ticker")["prob_up_8to9"])

    st.caption("Signal rule: LONG if prob ≥ 0.5, else SHORT. (Purely informational; not financial advice.)")
