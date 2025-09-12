import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["Close"].pct_change()
    df["ret_4"] = df["Close"].pct_change(4)
    df["ret_24"] = df["Close"].pct_change(24)

    rsi = RSIIndicator(df["Close"], window=14)
    df["rsi"] = rsi.rsi()

    ema20 = EMAIndicator(df["Close"], window=20)
    ema50 = EMAIndicator(df["Close"], window=50)
    sma200 = SMAIndicator(df["Close"], window=200)
    df["ema20"] = ema20.ema_indicator()
    df["ema50"] = ema50.ema_indicator()
    df["sma200"] = sma200.sma_indicator()

    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"]  = bb.bollinger_lband()
    df["bb_w"]    = (df["bb_high"] - df["bb_low"]) / df["Close"]

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr"] = atr.average_true_range()

    # Normalized volumes
    df["vol_ema20"] = df["Volume"].ewm(span=20).mean()
    df["vol_norm"] = df["Volume"] / (df["vol_ema20"] + 1e-9)

    # Drop initial NaNs from indicators
    df = df.dropna()

    return df

def feature_cols():
    return [
        "Open","High","Low","Close","Volume",
        "ret_1","ret_4","ret_24",
        "rsi","ema20","ema50","sma200",
        "macd","macd_signal","macd_hist",
        "bb_high","bb_low","bb_w","atr","vol_norm"
    ]
