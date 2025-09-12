# src/data_sheet.py
from __future__ import annotations
import json
from typing import List, Optional
import pandas as pd
import pytz
import gspread
from google.oauth2.service_account import Credentials

def _get_client_from_env() -> gspread.Client:
    """
    Reads a JSON Service Account key from env var GOOGLE_SERVICE_ACCOUNT_JSON.
    """
    raw = None
    try:
        import os
        raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        if not raw:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON env not set.")
        info = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Could not load service account JSON from env: {e!r}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def _read_sheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
    gc = _get_client_from_env()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)
    # get_all_records is simple & robust for modest data sizes
    rows = ws.get_all_records()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Timestamp","Ticker","Open","High","Low","Close","Volume"])
    # Normalize expected columns (case-insensitive)
    lcmap = {c.lower(): c for c in df.columns}
    need = ["timestamp","ticker","open","high","low","close","volume"]
    missing = [n for n in need if n not in lcmap]
    if missing:
        raise RuntimeError(f"Sheet is missing columns: {missing}. Expected: {need}")
    df = df[[lcmap[n] for n in need]]
    df.columns = ["Timestamp","Ticker","Open","High","Low","Close","Volume"]
    return df

def concat_pairs_sheet(
    tickers: List[str],
    start: str,
    end: Optional[str],
    tz_name: str = "Europe/London",
    sheet_id: str = "",
    worksheet: str = "fx_1h",
) -> pd.DataFrame:
    """
    Load all rows from the Google Sheet, parse Timestamp to tz-aware,
    filter by date range and tickers, and return OHLCV+Ticker indexed by time.
    """
    if not sheet_id:
        raise ValueError("Sheet id required")
    df = _read_sheet(sheet_id, worksheet)
    if df.empty:
        return pd.DataFrame()

    # Parse Timestamp; assume it's in Europe/London local time (as recommended)
    idx = pd.to_datetime(df["Timestamp"], errors="coerce")
    if idx.isna().any():
        df = df.loc[idx.notna()].copy()
        idx = pd.to_datetime(df["Timestamp"])
    # Localize to London then convert to desired tz (usually also London)
    ldn = pytz.timezone("Europe/London")
    idx = idx.dt.tz_localize(ldn, nonexistent="NaT", ambiguous="NaT")
    df = df.loc[idx.notna()].copy()
    idx = idx.dropna()
    df.index = idx
    if tz_name and tz_name != "Europe/London":
        df.index = df.index.tz_convert(pytz.timezone(tz_name))

    # Filter by date range
    start_ts = pd.Timestamp(start, tz=df.index.tz)
    end_ts = pd.Timestamp(end, tz=df.index.tz) if end else None
    mask = df.index >= start_ts
    if end_ts is not None:
        mask &= (df.index <= end_ts)
    df = df.loc[mask].copy()

    # Filter by tickers
    s = set([t.strip() for t in tickers])
    df = df[df["Ticker"].astype(str).str.strip().isin(s)].copy()

    # Ensure numeric cols
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    return df.sort_index()[["Open","High","Low","Close","Volume","Ticker"]]
