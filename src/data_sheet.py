# src/data_sheet.py
from __future__ import annotations

import os
import re
import json
from typing import Iterable, Sequence, Optional

import pandas as pd
import pytz
import gspread
from google.oauth2.service_account import Credentials

REQUIRED_COLS = [
    "Timestamp (London)", "Pair", "Open", "High", "Low", "Close", "Volume"
]

def _sheets_client() -> gspread.Client:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON not set. Add the service-account JSON "
            "to repo secrets and export it in the workflow env."
        )
    info = json.loads(raw)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def _extract_key(maybe_url: str) -> Optional[str]:
    if "docs.google.com/spreadsheets" in maybe_url:
        m = re.search(r"/d/([a-zA-Z0-9-_]{20,})/", maybe_url)
        return m.group(1) if m else None
    return maybe_url

def _open_sheet(gc: gspread.Client, sheet_id_or_url: str) -> gspread.Spreadsheet:
    key = _extract_key(sheet_id_or_url)
    if key:
        try:
            return gc.open_by_key(key)
        except gspread.SpreadsheetNotFound:
            pass
    try:
        return gc.open_by_url(sheet_id_or_url)
    except gspread.SpreadsheetNotFound as ex:
        raise RuntimeError(
            "Google Sheet not found.\n"
            "• config.yaml → data.sheet.id must be a valid key or full URL\n"
            "• Share the sheet with your service-account email (Viewer/Editor)\n"
            f"Original error: {ex}"
        ) from ex

def _read_tab_as_dataframe(sheet_id: str, worksheet: str) -> pd.DataFrame:
    gc = _sheets_client()
    sh = _open_sheet(gc, sheet_id)
    try:
        ws = sh.worksheet(worksheet)
    except gspread.WorksheetNotFound as ex:
        raise RuntimeError(
            f"Worksheet/tab '{worksheet}' not found. "
            f"Available tabs: {[w.title for w in sh.worksheets()]}"
        ) from ex

    rows = ws.get_all_values()
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLS)
    header, *data = rows
    return pd.DataFrame(data, columns=header)

def _coerce_and_filter(df: pd.DataFrame, start: str | None, end: str | None, tz_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # If the sheet accidentally has "Ticker" from earlier runs, rename to "Pair"
        if "Ticker" in missing and "Ticker" in df.columns:
            df = df.rename(columns={"Ticker": "Pair"})
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing expected columns in sheet: {missing}")

    df = df.copy()
    df["Timestamp (London)"] = pd.to_datetime(df["Timestamp (London)"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Timestamp (London)", "Pair", "Open", "High", "Low", "Close"])

    tz = pytz.timezone(tz_name)
    idx = pd.DatetimeIndex(df["Timestamp (London)"]).tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    )
    df.index = idx
    df = df.sort_index()

    if start:
        df = df[df.index >= pd.Timestamp(start, tz=tz)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=tz)]

    return df[["Pair", "Open", "High", "Low", "Close", "Volume"]]

def concat_pairs_sheet(
    pairs: Iterable[str],
    start: Optional[str],
    end: Optional[str],
    tz_name: str,
    sheet_id: str,
    worksheet: str,
) -> pd.DataFrame:
    raw = _read_tab_as_dataframe(sheet_id, worksheet)
    if raw.empty:
        return pd.DataFrame(columns=["Pair", "Open", "High", "Low", "Close", "Volume"])
    df = _coerce_and_filter(raw, start=start, end=end, tz_name=tz_name)
    want = [p.strip() for p in pairs if p and p.strip()]
    if want:
        df = df[df["Pair"].isin(want)]
    return df

def append_rows(
    sheet_id: str,
    worksheet: str,
    rows: Sequence[Sequence[object]],
    header: Optional[Sequence[str]] = None,
) -> None:
    if not rows:
        return
    gc = _sheets_client()
    sh = _open_sheet(gc, sheet_id)
    try:
        ws = sh.worksheet(worksheet)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet, rows=2000, cols=max(10, len(header or rows[0])))
    existing = ws.get_all_values()
    if not existing and header:
        ws.append_row(list(header))
    ws.append_rows([list(r) for r in rows], value_input_option="USER_ENTERED")
