from __future__ import annotations
import pandas as pd

def make_9to13_label(df_4h: pd.DataFrame) -> pd.Series:
    """
    Given a 4H OHLC frame indexed in Europe/London with a 'Close' column and 'Pair',
    return a binary label that is +1 if the 09:00→13:00 candle closed up vs its open,
    else 0. Rows not representing the 09:00 candle get NaN.
    """
    if df_4h.empty:
        return pd.Series([], dtype="float64", index=df_4h.index)

    # Keep only the candles that *start* at 09:00 (London).
    is_0900 = (df_4h.index.hour == 9) & (df_4h.index.minute == 0)
    df = df_4h.copy()
    df.loc[~is_0900, "label"] = float("nan")
    # For the 09→13 candle, label by Close vs Open.
    up = (df["Close"] - df["Open"]) > 0
    df.loc[is_0900, "label"] = up.astype(int)
    return df["label"]
