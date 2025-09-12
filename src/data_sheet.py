# src/data_sheet.py
from __future__ import annotations

import json
import os
from typing import List, Optional

import pandas as pd
import pytz
import gspread
from google.oauth2.service_account import Credentials


LONDON_TZ = pytz.timezone("Europe/London")
REQUIRED_COLS_LOWER = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]


# ----------------------------
# Auth / Sheets helpers
# ----------------------------
def _get_client_from_env() -> gspread.Client:
    """
    Build a gspread client from the GOOGLE_SERVICE_ACCOUNT_JSON env var
    (the full JSON key copied into a GitHub Actions secret).
    """
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not raw:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON environment variable is missing. "
            "Add it as a repository secret and expose it in your workflow's env."
        )
    try:
        info = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON: {e!r}")

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def _read_entire_sheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
    """
    Read all rows from a Google Sheet tab into a DataFrame.
    Assumes first row is headers.
    """
    if not sheet_id:
        raise ValueError("sheet_id is required")
    if not worksheet:
        raise ValueError("worksheet (tab name) is required")

    gc = _get_client_from_env()
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)

    # get_all_records() returns a list of dicts with headers from row 1
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    # If the sheet is empty, return a DataFrame with expected columns
    if df.empty:
        return pd.DataFrame(columns=[c.title() for c in REQUIRED_COLS_LOWER])
    return df


# ----------------------------
# Public API used by the repo
# ----------------------------
def concat_pairs_sheet(
    tickers: List[str],
    start: str,
    end: Optional[str],
    tz_name: str = "Europe/London",
    sheet_id: str = "",
    worksheet: str = "fx_1h",
) -> pd.DataFrame:
    """
    Load all rows from a Google Sheet tab, normalize, and filter to the
    requested date range and tickers. Returns a DataFrame indexed by time
    with columns: ['Open','High','Low','Close','Volume','Ticker'].

    Parameters
    ----------
    tickers : List[str]
        E.g. ["EURUSD=X","GBPUSD=X", ...] (must match 'Ticker' values in the sheet).
    start : str
        Inclusive start (e.g., "2024-01-01"). Assumed in tz_name.
    end : Optional[str]
        Inclusive end (e.g., "2024-12-31"). If None, uses 'now'.
    tz_name : str
        Target timezone for the index (default "Europe/London").
    sheet_id : str
        Google Sheet ID (the long string in the sheet URL).
    worksheet : str
        Tab name within the sheet (e.g., "fx_1h").
    """
    # --- Read raw data ---
    raw = _read_entire_sheet(sheet_id, worksheet)

    # --- Normalize headers case-insensitively ---
    # Build a lowercase -> actual-name map
    lcmap = {str(c).strip().lower(): c for c in raw.columns}
    missing = [c for c in REQUIRED_COLS_LOWER if c not in lcmap]
    if missing:
        raise RuntimeError(
            f"Sheet '{worksheet}' is missing columns: {missing}. "
            f"Expected at least {REQUIRED_COLS_LOWER} (case-insensitive)."
        )

    df = raw[[lcmap[c] for c in REQUIRED_COLS_LOWER]].copy()
    df.columns = [c.title() for c in REQUIRED_COLS_LOWER]  # "Timestamp","Ticker","Open",...

    # --- Parse timestamps and set index ---
    # We expect "Timestamp" to be in local London time (per your seeding step).
    idx = pd.to_datetime(df["Timestamp"], errors="coerce")
    if idx.isna().any():
        # Drop bad rows if any
        df = df.loc[idx.notna()].copy()
        idx = idx[idx.notna()]
    # Localize from naive London to tz-aware
    # If timestamps already include tz info, to_datetime will preserve it; guard both cases:
    if getattr(idx.dt, "tz", None) is None:
        idx = idx.dt.tz_localize(LONDON_TZ, nonexistent="NaT", ambiguous="NaT")
    # Drop rows that became NaT due to DST edge cases
    good = idx.notna()
    df = df.loc[good].copy()
    idx = idx[good]

    # Convert to requested tz (often also London)
    target_tz = pytz.timezone(tz_name) if tz_name else LONDON_TZ
    idx = idx.dt.tz_convert(target_tz)
    df.index = idx
    df.index.name = None  # downstream code often sets its own index name

    # --- Coerce numerics & fill volume ---
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = df["Volume"].fillna(0)

    # Drop rows missing core OHLC after coercion
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # --- Filter by date range (inclusive) ---
    start_ts = pd.Timestamp(start, tz=target_tz)
    end_ts = pd.Timestamp(end, tz=target_tz) if end else pd.Timestamp.now(tz=target_tz)
    m = (df.index >= start_ts) & (df.index <= end_ts)
    df = df.loc[m].copy()

    # --- Filter by tickers (exact string match after stripping) ---
    want = set(t.strip() for t in tickers if str(t).strip())
    if want:
        df["Ticker"] = df["Ticker"].astype(str).str.strip()
        df = df[df["Ticker"].isin(want)].copy()

    # --- Final shape / ordering ---
    if df.empty:
        # Return an empty frame with the expected columns in the expected order
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Ticker"]).sort_index()

    return df[["Open", "High", "Low", "Close", "Volume", "Ticker"]].sort_index()


# ----------------------------
# Optional: CLI smoke test
# ----------------------------
if __name__ == "__main__":
    """
    Run a quick local/CI smoke test:
      python -m src.data_sheet --sheet-id <ID> --worksheet fx_1h --start 2024-01-01 --end 2024-12-31 --tickers "EURUSD=X,GBPUSD=X"
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sheet-id", required=True, help="Google Sheet ID (from the URL)")
    p.add_argument("--worksheet", default="fx_1h", help="Worksheet/tab name")
    p.add_argument("--start", required=True, help="Inclusive start, e.g., 2024-01-01")
    p.add_argument("--end", default=None, help="Inclusive end, e.g., 2024-12-31 (optional)")
    p.add_argument("--tz", default="Europe/London", help="Target timezone for index")
    p.add_argument("--tickers", default="EURUSD=X", help="Comma-separated tickers to include")
    args = p.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    df = concat_pairs_sheet(
        tickers=tickers,
        start=args.start,
        end=args.end,
        tz_name=args.tz,
        sheet_id=args.sheet_id,
        worksheet=args.worksheet,
    )
    print("Rows:", len(df))
    print(df.head(5))
