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

# Expected columns in your Google Sheet tabs holding OHLC:
REQUIRED_COLS = [
    "Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume"
]

# ----------------------------
# Authentication / sheet open
# ----------------------------
def _sheets_client() -> gspread.Client:
    """
    Build a gspread client from the GOOGLE_SERVICE_ACCOUNT_JSON secret.
    """
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON not set. Add the service-account JSON "
            "to your repo secrets and expose it as an env var in the workflow."
        )
    info = json.loads(raw)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def _extract_key(maybe_url: str) -> Optional[str]:
    """
    If a full URL is provided, extract /d/<key>/; otherwise return the string.
    """
    if "docs.google.com/spreadsheets" in maybe_url:
        m = re.search(r"/d/([a-zA-Z0-9-_]{20,})/", maybe_url)
        return m.group(1) if m else None
    return maybe_url


def _open_sheet(gc: gspread.Client, sheet_id_or_url: str) -> gspread.Spreadsheet:
    """
    Try open_by_key first, then open_by_url, with friendly errors.
    """
    key = _extract_key(sheet_id_or_url)
    if key:
        try:
            return gc.open_by_key(key)
        except gspread.SpreadsheetNotFound:
            # fall through to URL attempt
            pass
    try:
        return gc.open_by_url(sheet_id_or_url)
    except gspread.SpreadsheetNotFound as ex:
        raise RuntimeError(
            "Google Sheet not found.\n"
            "Check that:\n"
            "  • config.yaml → data.sheet.id is the correct key or full URL\n"
            "  • The sheet is shared with your service-account email (Viewer/Editor)\n"
            "  • The sheet exists and you can open it in a browser\n"
            f"Original error: {ex}"
        ) from ex


# ----------------------------
# Reading helpers
# ----------------------------
def _read_tab_as_dataframe(sheet_id: str, worksheet: str) -> pd.DataFrame:
    """
    Read all values from a specific tab into a DataFrame (header row + data).
    """
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
    df = pd.DataFrame(data, columns=header)
    return df


def _coerce_and_filter(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    tz_name: str,
) -> pd.DataFrame:
    """
    Validate columns, coerce dtypes, localize to tz, filter by [start, end].
    """
    if df.empty:
        return df

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in sheet: {missing}")

    # Coerce types
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Pair"] = df["Pair"].astype(str)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Timestamp", "Pair", "Open", "High", "Low", "Close"]).copy()

    # Timezone handling
    tz = pytz.timezone(tz_name)
    # If the timestamps are naive local times (as per your sheet), localize them:
    idx = pd.DatetimeIndex(df["Timestamp"]).tz_localize(
        tz, nonexistent="shift_forward", ambiguous="NaT"
    )
    df.index = idx
    df = df.sort_index()

    # Date filters
    if start:
        df = df[df.index >= pd.Timestamp(start, tz=tz)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=tz)]

    # Standardize the column used downstream
    df = df.rename(columns={"Pair": "Ticker"})
    return df[["Ticker", "Open", "High", "Low", "Close", "Volume"]]


def concat_pairs_sheet(
    tickers: Iterable[str],
    start: Optional[str],
    end: Optional[str],
    tz_name: str,
    sheet_id: str,
    worksheet: str,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame of OHLC rows for the requested tickers from a sheet tab.
    Columns returned: ['Ticker','Open','High','Low','Close','Volume']
    Index: timezone-aware DatetimeIndex in tz_name (e.g., Europe/London)
    """
    raw = _read_tab_as_dataframe(sheet_id, worksheet)
    if raw.empty:
        return pd.DataFrame(columns=["Ticker", "Open", "High", "Low", "Close", "Volume"])

    df = _coerce_and_filter(raw, start=start, end=end, tz_name=tz_name)
    wanted = [t.strip() for t in tickers if t and t.strip()]
    if wanted:
        df = df[df["Ticker"].isin(wanted)]
    return df


# ----------------------------
# Writing helpers (append)
# ----------------------------
def append_rows(
    sheet_id: str,
    worksheet: str,
    rows: Sequence[Sequence[object]],
    header: Optional[Sequence[str]] = None,
) -> None:
    """
    Append rows to a tab. If the tab is empty and a header is provided, it will write
    the header first, then the rows.
    """
    if not rows:
        return
    gc = _sheets_client()
    sh = _open_sheet(gc, sheet_id)
    try:
        ws = sh.worksheet(worksheet)
    except gspread.WorksheetNotFound:
        # Create the worksheet if it doesn't exist (requires Editor permission).
        ws = sh.add_worksheet(title=worksheet, rows=2000, cols=max(10, len(header or rows[0])))

    existing = ws.get_all_values()
    if not existing and header:
        ws.append_row(list(header))
    ws.append_rows([list(r) for r in rows], value_input_option="USER_ENTERED")
