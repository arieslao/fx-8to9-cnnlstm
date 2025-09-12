# src/data_dk.py
from datetime import datetime
import pandas as pd
import pytz

# pip install dukascopy-python
from dukascopy_python.client import DukascopyClient

_LDN = pytz.timezone("Europe/London")

def _to_dk_symbol(yahoo_fx: str) -> str:
    """
    Convert Yahoo FX like 'EURUSD=X' -> 'EURUSD' for Dukascopy.
    """
    return yahoo_fx.replace("=X", "").upper()

def fetch_fx_hourly_dk(ticker: str, start: str, end: str | None, tz_name: str = "Europe/London") -> pd.DataFrame:
    """
    Fetch 1-hour candles [bid mid] from Dukascopy and align index to London time.
    start/end are ISO-like strings in *local* time window; we convert to UTC ranges internally.
    """
    sym = _to_dk_symbol(ticker)

    # Parse start/end as timezone-aware London datetimes, then convert to UTC for the API
    start_dt = pd.Timestamp(start, tz=_LDN)
    end_dt = pd.Timestamp.now(tz=_LDN) if end is None else pd.Timestamp(end, tz=_LDN)

    # Dukascopy client prefers UTC
    start_utc = start_dt.tz_convert("UTC").to_pydatetime()
    end_utc = end_dt.tz_convert("UTC").to_pydatetime()

    # Download hour bars (OHLC, volume). Dukascopy provides bid/ask; we’ll use mid.
    # The client returns a DataFrame with columns like: ['open', 'high', 'low', 'close', 'volume', 'time']
    cli = DukascopyClient()
    df = cli.candles(
        instrument=sym,
        time_frame="H1",         # 1-hour
        from_time=start_utc,
        to_time=end_utc,
        price_type="BID_ASK"     # get both; we'll compute mid
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure datetime index in London timezone
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(_LDN)
    df = df.set_index("time").sort_index()

    # If both bid/ask present, compute mid; otherwise keep what we have
    # Some clients return columns like openBid/openAsk/... Adjust robustly.
    def _mid(a, b): return (a + b) / 2.0

    col_map = {
        "open": ["openBid", "openAsk"],
        "high": ["highBid", "highAsk"],
        "low":  ["lowBid",  "lowAsk"],
        "close":["closeBid","closeAsk"],
        "volume":["volume"]  # volume may be present as a single column
    }

    def _get(name):
        opts = col_map[name]
        if len(opts) == 1 and opts[0] in df.columns:
            return df[opts[0]]
        if opts[0] in df.columns and opts[1] in df.columns:
            return _mid(df[opts[0]], df[opts[1]])
        # Fallback to any similarly named column
        for c in df.columns:
            if name in c.lower():
                return df[c]
        return pd.Series(index=df.index, dtype="float64")

    out = pd.DataFrame({
        "Open":  _get("open"),
        "High":  _get("high"),
        "Low":   _get("low"),
        "Close": _get("close"),
        "Volume": _get("volume").fillna(0.0),
    }).dropna(how="all")

    out["Ticker"] = ticker  # keep original Yahoo-style name
    return out

def concat_pairs_dk(tickers, start, end, tz_name="Europe/London"):
    frames = []
    for t in tickers:
        df = fetch_fx_hourly_dk(t, start, end, tz_name=tz_name)
        if not df.empty:
            frames.append(df)
        else:
            print(f"[INFO] Dukascopy returned no data for {t}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()
