from __future__ import annotations

import os
import re
import json
from typing import List

import pandas as pd
import numpy as np
import pytz

import gspread
from google.oauth2.service_account import Credentials


def _service_account_client() -> gspread.Client:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is not set. Add the full JSON to repo Secrets."
        )
    info = json.loads(raw)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


_GS_KEY_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")

def _extract_key(sheet_id_or_url: str) -> str:
    """
    Accepts either the raw key (long random string) or a full Google Sheets URL.
    Returns just the key.
    """
    m = _GS_KEY_RE.search(sheet_id_or_url)
    return m.group(1) if m else sheet_id_or_url


def _open_sheet(gc: gspread.Client, sheet_id_or_url: str):
    key = _extract_key(sheet_id_or_url)
    try:
        return gc.open_by_key(key)
    except gspread.exceptions.SpreadsheetNotFound as ex:
        raise RuntimeError(
            "Google Sheet not found (404). "
            "Double-check the Sheet ID/URL and make sure the Sheet is shared with your "
            "service account email (see the smoke step output)."
        ) from ex


def _worksheet_to_dataframe(ws) -> pd.DataFrame:
    """
    Read the entire worksheet with gspread (no gspread_dataframe dependency)
    and convert it to a pandas DataFrame using the first row as headers.
    """
    values = ws.get_all_values()  # list[list[str]]
    if not values:
        return pd.DataFrame()

    header = values[0]
    rows = values[1:]

    # Build DataFrame; keep empty strings as NaN for numeric coercion later
    df = pd.DataFrame(rows, columns=header)
    df.replace({"": pd.NA}, inplace=True)
    return df


def _read_entire_sheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
    gc = _service_account_client()
    sh = _open_sheet(gc, sheet_id)
    try:
        ws = sh.worksheet(worksheet)
    except gspread.exceptions.WorksheetNotFound as ex:
        raise RuntimeError(
            f"Worksheet/tab '{worksheet}' not found. Check the tab name in config.yaml."
        ) from ex

    df = _worksheet_to_dataframe(ws)
    # Drop rows that are entirely NaN.
    df = df.dropna(how="all")
    return df


def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns in the sheet: {missing}. "
            "Expected header: ['Timestamp (London)', 'Pair', 'Open', 'High', 'Low', 'Close', 'Volume']"
        )


def _coerce_schema(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    # Rename any common variants to our canonical names
    rename_map = {
        "Ticker": "Pair",
        "timestamp": "Timestamp (London)",
        "Timestamp": "Timestamp (London)",
        "Time": "Timestamp (London)",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    _require_columns(
        df, ["Timestamp (London)", "Pair", "Open", "High", "Low", "Close", "Volume"]
    )

    # Parse timestamp and make tz-aware (Europe/London)
    tz = pytz.timezone(tz_name)
    ts = pd.to_datetime(df["Timestamp (London)"], errors="coerce", utc=False)
    if ts.dt.tz is None:
        # localize naive timestamps to London
        ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        # if already tz-aware, convert to London
        ts = ts.dt.tz_convert(tz)
    df["ts"] = ts

    # Basic numeric coercion
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts", "Pair", "Open", "High", "Low", "Close"])
    df = df.sort_values(["Pair", "ts"]).reset_index(drop=True)

    # Final index for convenience
    df = df.set_index("ts")
    return df


def concat_pairs_sheet(
    pairs: list[str],
    start: str,
    end: str,
    tz_name: str,
    sheet_id: str,
    worksheet: str,
) -> pd.DataFrame:
    """
    Read all rows from the given worksheet, coerce schema, filter to pairs and date range,
    and return an index on 'ts' with columns: Pair, Open, High, Low, Close, Volume
    """
    raw = _read_entire_sheet(sheet_id, worksheet)
    df = _coerce_schema(raw, tz_name)

    # Filter to date range and selected pairs
    start_ts = pd.Timestamp(start, tz=tz_name)
    end_ts = pd.Timestamp(end, tz=tz_name)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df = df[df["Pair"].isin([p.strip() for p in pairs])]
    return df[["Pair", "Open", "High", "Low", "Close", "Volume"]].copy()
