# src/smoke.py
import os, sys, json
import pandas as pd

print("### PYTHON VERSION", sys.version)
print("### FX_TICKERS =", os.getenv("FX_TICKERS"))

# verify dukascopy is importable
try:
    import dukascopy_python as dk
    import dukascopy_python.instruments as inst
    print("### dukascopy_python import OK")
except Exception as e:
    print("!!! dukascopy_python import FAILED:", repr(e))
    sys.exit(2)

# verify our code imports
try:
    from src.utils import load_config
    from src.data_dk import concat_pairs_dk
    print("### repo imports OK")
except Exception as e:
    print("!!! repo imports FAILED:", repr(e))
    sys.exit(3)

cfg = load_config()
pairs = os.getenv("FX_TICKERS", "EURUSD=X").split(",")
pairs = [p.strip() for p in pairs if p.strip()]
print("### using pairs:", pairs)

# fetch a tiny window (~48h) just to prove access & schema
end = pd.Timestamp.now(tz=cfg["data"]["timezone"])
start = (end - pd.Timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")

df = concat_pairs_dk(pairs[:1], start, None, tz_name=cfg["data"]["timezone"])
print("### fetched rows:", len(df))
print("### columns:", list(df.columns))
print(df.tail(3))

# minimal schema check
required = {"Open","High","Low","Close","Volume","Ticker"}
missing = required - set(df.columns)
if missing:
    print("!!! missing columns:", missing)
    sys.exit(4)

print("### SMOKE OK")
