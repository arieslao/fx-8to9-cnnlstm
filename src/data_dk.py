# src/data_dk.py
from __future__ import annotations
from typing import List, Optional
import pandas as pd
import pytz

# Import defensively
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
            "dukascopy-python not importable. Ensure requirements.txt has dukascopy-python==4.0.1. "
            f"Original error: {_DK_IMPORT_ERROR!r}"
        )

def _normalize_symbol(t: str) -> str:
    return t.strip().upper().replace("=X", "")  # "EURUSD=X" -> "EURUSD"

def _instrument_for(pair6: str):
    base, quote = pair6[:3], pair6[3:]
    name = f"INSTRUMENT_FX_MAJORS_{base}_{quote}"
    if inst is not None and hasattr(inst, name):
        return getattr(inst, name)
    return pair6  # many APIs accept "EURUSD" directly

def _call_dk_api(instrument, start_utc, end_utc) -> pd.DataFrame:
    """
    Try several known API shapes in dukascopy-python across versions:
      1) fetch(instrument, interval, offer_side, start, end)               (positional)
      2) fetch(instrument, interval, start, end)                           (positional, no offer)
      3) fetch(instrument=..., interval=..., date_from=..., date_to=..., offer_side=...) (kwargs)
      4) get_price_history(..., timeframe='1H')
      5) get_data(..., timeframe='1H')
    """
    if hasattr(dk, "fetch"):
        interval = getattr(dk, "INTERVAL_HOUR_1", None)
        offer    = getattr(dk, "OFFER_SIDE_BID", None)

        # (1) full positional
        try:
            return dk.fetch(instrument, interval, offer, start_utc, end_utc)
        except TypeError:
            pass

        # (2) positional without offer_side
        try:
            return dk.fetch(instrument, interval, start_utc, end_utc)
        except TypeError:
            pass

        # (3) kwargs style (older/newer alt)
        try:
            kwargs = {}
            if interval is not None: kwargs["interval"] = interval
            if offer is not None:    kwargs["offer_side"] = offer
            return dk.fetch(instrument=instrument, date_from=start_utc, date_to=end_utc, **kwargs)
        except TypeError:
            pass

    # (4) alternate API name
    if hasattr(dk, "get_price_history"):
        return dk.get_price_history(instrument=instrument, start=start_utc, end=end_utc, timeframe="1H")

    # (5) alternate API name
    if hasattr(dk, "get_data"):
        return dk.get_data(instrument=instrument, date_from=start_utc, date_to=end_utc, timeframe="1H")

    raise AttributeError(
        "dukascopy-python on this runner exposes neither a compatible fetch() nor get_price_history()/get_data()."
    )

def _normalize_df(df: pd.DataFrame, ticker: str, tz_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","Ticker"])

    # index → London tz
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = "time" if "time" in df.columns else None
        if tcol:
            df = df.set_index(pd.to_datetime(df[tcol], utc=True))
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz_name)

    # columns → Title case used by pipeline
    rename = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc in ("open","high","low","close","volume"):
            rename[c] = lc.title()
    df = df.rename(columns=rename)
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["Ticker"] = ticker
    return df[["Open","High","Low","Close","Volume","Ticker"]].sort_index()

def fetch_pair_dk(ticker: str, start: str, end: Optional[str], tz_name: str) -> pd.DataFrame:
    _require_dk()
    pair6 = _normalize_symbol(ticker)
    instrument = _instrument_for(pair6)

    # convert London-local strings → UTC datetimes for API
    start_utc = pd.Timestamp(start, tz=tz_name).tz_convert("UTC").to_pydatetime()
    end_ts = pd.Timestamp(end, tz=tz_name) if end else pd.Timestamp.now(tz=tz_name)
    end_utc = end_ts.tz_convert("UTC").to_pydatetime()

    raw = _call_dk_api(instrument, start_utc, end_utc)
    return _normalize_df(raw, ticker, tz_name)

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
