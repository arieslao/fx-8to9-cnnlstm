from __future__ import annotations

import os
import json
import sys
import pandas as pd

# Local imports
from src.data_sheet import concat_pairs_sheet

def _load_config() -> dict:
    import yaml
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def _pairs_from_env() -> list[str]:
    raw = os.getenv("FX_PAIRS") or os.getenv("FX_TICKERS") or "EURUSD"
    return [p.strip() for p in raw.split(",") if p.strip()]

def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main() -> None:
    _print_header("[SMOKE] Google Sheets read")

    # 0) Confirm the service account JSON exists and show which email to share with
    svc = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not svc:
        sys.exit("GOOGLE_SERVICE_ACCOUNT_JSON is not set. Add the full JSON to repo Secrets and export it in the workflow env.")
    try:
        info = json.loads(svc)
        sa_email = info.get("client_email", "<missing>")
        print("Service account:", sa_email)
    except Exception as e:
        sys.exit(f"Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON: {e}")

    # 1) Load config values
    cfg = _load_config()
    try:
        sheet_id = cfg["data"]["sheet"]["id"]
        worksheet = cfg["data"]["sheet"]["worksheet"]
        tz_name = cfg["data"]["timezone"]
        start = cfg["data"]["train_start"]
        end = cfg["data"]["train_end"]
    except KeyError as e:
        sys.exit(f"config.yaml is missing a required key: {e}")

    print("Sheet ID (key or URL):", sheet_id)
    print("Worksheet (tab):      ", worksheet)
    print("Timezone:             ", tz_name)
    print("Train range:          ", start, "→", end)

    # 2) Select pairs
    pairs = _pairs_from_env()
    print("Pairs:", pairs)

    # 3) Try reading a small slice from the sheet
    try:
        df = concat_pairs_sheet(
            pairs=pairs,
            start=start,
            end=end,
            tz_name=tz_name,
            sheet_id=sheet_id,
            worksheet=worksheet,
        )
    except Exception as e:
        # Make errors very explicit for first-time setup
        print("\n[ERROR] Failed to open/read Google Sheet.")
        print("Common causes:")
        print("  • Wrong Sheet ID (use the long key between /d/ and /edit, or paste the full URL)")
        print("  • Sheet not shared with the service account above (Viewer or Editor)")
        print("  • Worksheet (tab) name typo")
        print("  • Org policy blocking external sharing")
        raise

    # 4) Basic schema sanity
    expected = ["Pair", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print("\n[ERROR] Missing columns:", missing)
        print("Sheet tab must have header exactly:")
        print("  Timestamp (London) | Pair | Open | High | Low | Close | Volume")
        sys.exit(1)

    # 5) Report and soft sample
    print("\nRows read:", len(df))
    if len(df):
        # show distinct pairs and a peek at the last few rows
        uniq = sorted(df["Pair"].unique())
        print("Distinct pairs:", uniq)
        print("\nTail(5):")
        with pd.option_context("display.max_columns", 20, "display.width", 160):
            print(df.tail(5))

    # 6) Final guard
    if df.empty:
        sys.exit("No rows read — check Sheet ID/tab share & the date range in config.yaml.")

    print("\n[SMOKE OK] Google Sheets data is accessible and correctly formatted.")

if __name__ == "__main__":
    main()
