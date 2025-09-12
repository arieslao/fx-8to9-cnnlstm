import pandas as pd
import yfinance as yf
import pytz

def fetch_fx_hourly(ticker, start, end, interval="1h", tz_name="Europe/London"):
    # yfinance returns UTC; we’ll localize to London
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)  # Open,High,Low,Close,Volume
    df.index = df.index.tz_localize("UTC").tz_convert(pytz.timezone(tz_name))
    df = df[["Open","High","Low","Close","Volume"]]
    df["Ticker"] = ticker
    return df

def concat_pairs(tickers, start, end, interval="1h", tz_name="Europe/London"):
    frames = []
    for t in tickers:
        df = fetch_fx_hourly(t, start, end, interval, tz_name)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()
