# src/data_dk.py
from __future__ import annotations

import pandas as pd
import pytz
from typing import List, Optional
from datetime import datetime

import dukascopy_python as dk
import dukascopy_python.instruments as inst

_LDN = pytz.timezone("Europe/London")
_UTC = pytz.UTC

def _normalize_pair(yahoo_fx: str) -> str:
    """
    Convert Yahoo FX like 'EURUSD=X' -> 'EURUSD' for Dukascopy instruments.
    """
    return yahoo_fx.strip().upper().replace("=X", "")

def _instrument_from_pair(pair6: str):
    """
    Map 'EURUSD' -> instruments constant 'INSTRUMENT_FX_MAJORS_EUR_USD'.
    (Extend here if you later fetch minors/exotics.)
    """
    if len(pair6) != 6:
        raise ValueError(f"Unexpected pair format: {pair6}")
    base, quote = pair6[:3], pair6[3:]
    name = f"INSTRUMENT_FX_MAJORS_{base}_{quote}"
    if not hasattr(inst, name):
        raise KeyError(f"dukascopy_python.instruments missing {name}")
    return getattr(inst, name)

def _dk_interval_1h():
    return dk.INTERVAL_HOUR_1

def _dk_offer_side():
    # Use BID side for consistency; mid is not directly exposed here.
    # For most modeling use-cases, BID or ASK will be fine; keep constant.
    return dk.OFFER_SIDE_BID

def _parse_london(dt_str: Optional[str]) -> Optional[datetime]:
    if dt_str is None:
        return None
    ts = pd.Timestamp(dt_str, tz=_LDN)
    return ts.to_pydatetime()

def _to_utc(dt: datetime) -> datetime:
    return dt.astimezone(_UTC)

def _fetch_one(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    pair6 = _normalize_pair(ticker)
    instrument = _instrument_from_pair(pair6)

    start_ldn = _parse_london(start)
    end_ldn = _parse_london(end) if end else pd.Timestamp.now(tz=_LDN).to_pydatetime()

    start_utc = _to_utc(start_ldn)
    end_utc = _to_utc(end_ldn)

    df = dk.fetch(
        instrument=instrument,
        interval=_dk_interval_1h(),
        offer_side=_dk_offer_side(),
        date_from=start_utc,
        date_to=end_utc,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure tz-aware index and convert to London time
    if df.index.tz is None:
        df.index = df.index.tz_localize(_UTC)
    df.index = df.index.tz_convert(_LDN)

    # Keep consistent schema with the Yahoo path
    # dukascopy_python returns columns: open, high, low, close, volume
    cols = ["open", "high", "low", "close", "volume"]
    keep = [c for c in cols if c in df.columns]
    df = df[keep].copy()

    # Rename to title-case like the rest of the pipeline expects
    rename_map = {c: c.title() for c in keep}
    df = df.rename(columns=rename_map)

    # Ensure all expected columns exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["Ticker"] = ticker  # keep the original Yahoo-style symbol for downstream code
    return df[["Open", "High", "Low", "Close", "Volume", "Ticker"]]

def concat_pairs_dk(tickers: List[str], start: str, end: Optional[str], tz_name: str = "Europe/London") -> pd.DataFrame:
    """
    Match signature used elsewhere: returns a single DataFrame with OHLCV+Ticker in London tz.
    """
    parts = []
    for t in tickers:
        try:
            df = _fetch_one(t, start, end)
            if not df.empty:
                parts.append(df)
            else:
                print(f"[INFO] Dukascopy returned no data for {t}")
        except Exception as e:
            print(f"[WARN] Failed to fetch {t}: {e}")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    # Index is already London tz; no need to re-convert based on tz_name, but keep param for parity.
    return out
