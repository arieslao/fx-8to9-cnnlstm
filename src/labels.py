import pandas as pd

def make_8to9_label(df: pd.DataFrame, start_hour=8, end_hour=9, threshold=0.0) -> pd.DataFrame:
    """
    Label each day for each ticker: +1 if Close@09:00 > Close@08:00, else 0.
    Keep only the row at 08:00 (the decision time), with the label attached.
    """
    df = df.copy()
    df["date"] = df.index.date
    df["hour"] = df.index.hour

    # Get 08:00 and 09:00 rows per date/ticker
    eight = df[df["hour"] == start_hour].copy()
    nine  = df[df["hour"] == end_hour].copy()

    key_cols = ["Ticker","date"]
    eight = eight.reset_index().rename(columns={"index":"ts"})
    nine  = nine.reset_index().rename(columns={"index":"ts"})

    merged = eight.merge(
        nine[["Ticker","date","Close"]].rename(columns={"Close":"close_9"}),
        on=key_cols, how="inner"
    )

    merged = merged.rename(columns={"Close":"close_8"})
    merged["label_raw"] = merged["close_9"] - merged["close_8"]
    merged["label"] = (merged["label_raw"] > threshold).astype(int)

    # Use the 08:00 timestamp as the sample time
    merged = merged.set_index("ts").sort_index()

    # Keep a compact frame for modeling
    keep_cols = [
        "Ticker","date","close_8","close_9","label"
    ]
    return merged[keep_cols]
