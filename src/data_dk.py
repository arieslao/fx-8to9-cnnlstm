# src/data_dk.py
from __future__ import annotations
from typing import List, Optional
import pandas as pd
import pytz

# Import dukascopy lib defensively
try:
    import dukascopy_python as dk
    import dukascopy_python.instruments as inst
except Exception as e:
    dk = None
    inst = None
    _DK_IMPORT_ERROR = e

_LDN = pytz.timezone("Europe/London")

def _require_dk():
    if dk is None:
        raise ImportError(
            "dukascopy-python is not importable. Ensure requirements.txt contains "
            "dukascopy-python==4.0.1. Original error: %r" % (_DK_IMPORT_ERROR,)
        )

def _normalize_symbol(t: str) -> str:
    # "EURUSD=X" -> "EURUSD"
    return t.strip().upper().replace("=X", "")

def _instrument_for(pair6: str):
    """
    Try to resolve 'EURUSD' -> dukascopy instrument constant, e.g.
    dukascopy_python.instruments.INSTRUMENT_FX_MAJORS_EUR_USD
    Falls back to raw 'EURUSD' if the constant doesn't exist.
    """
    base, quote = pair6[:3], pair6[3:]
    if inst is not None:
        name = f"INSTRUMENT_FX_MAJORS_{base}_{quote}"
        if hasattr(inst, name):
            return getattr(inst, name)
    # Fallback: many APIs accept plain 'EURUSD'
    return pair6

def _call_dk_api(instrument, start_utc, end_utc) -> pd.DataFrame:
    """
    Call whichever entry point exists in this dukascopy-python build,
    returning a DataFrame with columns at least [open, high, low, close, volume].
    """
    # 1) Preferred: dk.fetch(...)
    if hasattr(dk, "fetch"):
        # Try common constant names; if missing, pass None where allowed
        interval = getattr(dk, "INTERVAL_HOUR_1", None)
        offer   = getattr(dk, "OFFER_SIDE_BID", None)
        args = {}
        if interval is not None: args["interval"] = interval
        if offer   is not None: args["offer_side"] = offer
        return dk.fetch(instrument=instrument, date_from=start_utc, date_to=end_utc, **args)

    # 2) Alternate: dk.get_price_history(...)
    if hasattr(dk, "get_price_history"):
        return dk.get_price_history(
            instrument=instrument,
            start=start_utc, end=end_utc,
            timeframe="1H"
        )

    # 3) Alternate: dk.get_data(...)
    if hasattr(dk, "get_data"):
        return dk.get_data(
            instrument=instrument,
            date_from=start_utc, date_to=end_utc,
            timeframe="1H"
        )

    raise AttributeError(
        "Unsupported dukascopy-python API on runner. Neither fetch(), "
        "get_price_history(), nor get_data() is available."
    )

def _normalize_df(df: pd.DataFrame, ticker: str, tz_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","Ticker"])

    # Ensure datetime index -> London tz
    if not isinstance(df.index, pd.DatetimeIndex):
        # Some variants return a 'time' column
        time_col = "time" if "time" in df.columns else None
        if time_col:
            df = df.set_index(pd.to_datetime(df[time_col], utc=True))
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz_name or _LDN)

    # Normalize column names to title case used by pipeline
    rename = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc in ("open","high","low","close","volume"):
            rename[c] = lc.title()
    df = df.rename(columns=rename)

    # Ensure required columns exist
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["Ticker"] = ticker
    return df[["Open","High","Low","Close","Volume","Ticker"]].sort_index()

def fetch_pair_dk(ticker: str, start: str, end: Optional[str], tz_name: str) -> pd.DataFrame:
    _require_dk()
    pair6 = _normalize_symbol(ticker)
    instrument = _instrument_for(pair6)

    # Convert London-local strings to UTC datetimes for the API
    start_utc = pd.Timestamp(start, tz=tz_name or _LDN).tz_convert("UTC").to_pydatetime()
    end_ts = pd.Timestamp(end, tz=tz_name or _LDN) if end else pd.Timestamp.now(tz=tz_name or _LDN)
    end_utc = end_ts.tz_convert("UTC").to_pydatetime()

    raw = _call_dk_api(instrument, start_utc, end_utc)
    return _normalize_df(raw, ticker, tz_name or "Europe/London")

def concat_pairs_dk(tickers: List[str], start: str, end: Optional[str], tz_name: str = "Europe/London") -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            df = fetch_pair_dk(t, start, end, tz_name)
            if not df.empty:
                frames.append(df)
            else:
                print(f"[INFO] Dukascopy returned empty for {t}")
        except Exception as e:
            print(f"[WARN] Dukascopy fetch failed for {t}: {e!r}")
    if not frames:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","Ticker"])
    return pd.concat(frames).sort_index()
