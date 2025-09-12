# src/data_dk.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import pytz

import dukascopy_python as dk  # <- correct package import
import dukascopy_python.instruments as inst  # constants live here
# Docs: https://pypi.org/project/dukascopy-python/ (usage shows top-level API and instruments) 

LONDON_TZ = pytz.timezone("Europe/London")
UTC = pytz.UTC

# Map Yahoo-style tickers like "EURUSD=X" -> "EURUSD"
def _normalize_pair(symbol: str) -> str:
    s = symbol.strip().upper()
    return s.replace("=X", "")

def _instrument_from_pair(pair6: str):
    """
    Convert 'EURUSD' -> instruments constant name like
    INSTRUMENT_FX_MAJORS_EUR_USD
    """
    if len(pair6) != 6:
        raise ValueError(f"Unexpected pair format: {pair6}")

    base, quote = pair6[:3], pair6[3:]
    name = f"INSTRUMENT_FX_MAJORS_{base}_{quote}"
    if not hasattr(inst, name):
        # Fallback for non-major combinations if needed (extend later)
        raise KeyError(f"dukascopy_python.instruments missing {name}")
    return getattr(inst, name)

@dataclass
class FetchConfig:
    start: datetime
    end: datetime
    interval: str = "1h"       # we’ll map to dk.INTERVAL_HOUR_1
    offer_side: str = "BID"    # BID or ASK

def _dk_interval(interval: str):
    interval = interval.lower()
    if interval == "1h" or interval == "1hour" or interval == "hour":
        return dk.INTERVAL_HOUR_1
    # Extend if you ever need other granularities
    raise ValueError(f"Unsupported interval: {interval}")

def _dk_offer(offer_side: str):
    offer_side = offer_side.upper()
    if offer_side == "BID":
        return dk.OFFER_SIDE_BID
    if offer_side == "ASK":
        return dk.OFFER_SIDE_ASK
    raise ValueError(f"Unsupported offer_side: {offer_side}")

def fetch_one_pair(pair: str, cfg: FetchConfig) -> pd.DataFrame:
    """
    Fetch OHLCV for a single pair between cfg.start and cfg.end (UTC),
    return a tz-aware London-indexed DataFrame with columns:
    [open, high, low, close, volume] and a 'pair' column.
    """
    pair6 = _normalize_pair(pair)
    instrument = _instrument_from_pair(pair6)
    interval = _dk_interval(cfg.interval)
    offer = _dk_offer(cfg.offer_side)

    # dukascopy_python expects naive/aware datetimes; we pass UTC-aware
    start_utc = cfg.start.astimezone(UTC)
    end_utc = cfg.end.astimezone(UTC)

    # Historical fetch (returns a pandas DataFrame; see PyPI usage)
    df = dk.fetch(instrument, interval, offer, start_utc, end_utc)

    # Expect columns for non-tick interval: open, high, low, close, volume (per docs)
    # Ensure index is tz-aware & convert to London time
    if df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    df.index = df.index.tz_convert(LONDON_TZ)

    # Keep a clean schema
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df["pair"] = pair6
    return df

def concat_pairs_dk(pairs: List[str], cfg: FetchConfig) -> Dict[str, pd.DataFrame]:
    """
    Fetch and return a dict { 'EURUSD': df, ... } for given pairs.
    """
    out: Dict[str, pd.DataFrame] = {}
    for p in pairs:
        try:
            d = fetch_one_pair(p, cfg)
            out[_normalize_pair(p)] = d
        except Exception as e:
            # Let the caller decide how to handle missing instruments/data
            raise RuntimeError(f"Failed to fetch {p}: {e}") from e
    return out

def concat_pairs_frame(pairs: List[str], cfg: FetchConfig) -> pd.DataFrame:
    """
    Convenience: return one concatenated DataFrame with a 'pair' column.
    """
    parts = []
    for p in pairs:
        parts.append(fetch_one_pair(p, cfg))
    if not parts:
        raise RuntimeError("No data fetched for any pair")
    return pd.concat(parts).sort_index()
