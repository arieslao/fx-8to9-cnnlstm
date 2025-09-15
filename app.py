import os
import pandas as pd
import streamlit as st
from src.data_sheet import _sheets_client, _open_sheet

st.set_page_config(page_title="FX 4H – 09→13 predictions", layout="wide")

SHEET_ID = st.secrets.get("SHEET_ID") or os.getenv("SHEET_ID")
TAB_DATA = "fx_4h_data"
TAB_PREDS = "preds_8to9"

if not SHEET_ID:
    st.error("Set SHEET_ID (Streamlit secret or env).")
    st.stop()

gc = _sheets_client()
sh = _open_sheet(gc, SHEET_ID)

def read_tab(tab: str) -> pd.DataFrame:
    rows = sh.worksheet(tab).get_all_values()
    if not rows: return pd.DataFrame()
    header, *data = rows
    return pd.DataFrame(data, columns=header)

st.header("Predictions – 09→13 London")
preds = read_tab(TAB_PREDS)
data = read_tab(TAB_DATA)

if not preds.empty:
    preds["proba_up"] = pd.to_numeric(preds["proba_up"], errors="coerce")
    st.dataframe(preds.tail(50), use_container_width=True)
else:
    st.info("No predictions yet.")

if not data.empty:
    st.subheader("Latest 4H OHLC (fx_4h_data)")
    st.dataframe(data.tail(50), use_container_width=True)
