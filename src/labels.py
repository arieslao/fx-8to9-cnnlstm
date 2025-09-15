# src/labels.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
import pandas as pd
import numpy as np


@dataclass
class SnapConfig:
    desired_start_hour: int = 9        # London 09:00
    desired_end_hour: int = 13         # London 13:00
    duration_hours: int = 4            # 4h window
    snap_tolerance_hours: float = 2.1  # max distance to snap a target time to a bar
    trading_days_only: bool = True
    ticker_col: str = "Ticker"
    price_col: str = "Close"


def _is_business_day(ts: pd.Timestamp) -> bool:
    return ts.dayofweek < 5


def _nearest_index_time(idx: pd.DatetimeIndex, target: pd.Timestamp, tolerance: pd.Timedelta) -> Optional[pd.Timestamp]:
    """Return the nearest timestamp in idx to target within tolerance, else None."""
    if len(idx) == 0:
        return None
    # searchsorted gives insertion point; check neighbors for min |diff|
    pos = idx.searchsorted(target)
    best_ts, best_diff = None, None
    for j in (pos - 1, pos, pos + 1):
        if 0 <= j < len(idx):
            ts = idx[j]
            diff = abs(ts - target)
            if best_diff is None or diff < best_diff:
                best_ts, best_diff = ts, diff
    if best_ts is not None and best_diff <= tolerance:
        return best_ts
    return None


def _labels_for_ticker(
    df_tkr: pd.DataFrame,
    cfg: SnapConfig,
) -> pd.DataFrame:
    """
    Build labels for one ticker by snapping the desired daily 09:00->13:00
    window to the closest available 4h bars in df_tkr.index.
    """
    out_rows = []
    idx = df_tkr.index

    # datelike iteration: use unique normalized dates present in the data
    # to avoid scanning weekends / gaps unnecessarily
    dates: Iterable[pd.Timestamp] = pd.to_datetime(pd.Series(idx.normalize().unique()))
    dates = dates.sort_values()

    tol = pd.Timedelta(hours=cfg.snap_tolerance_hours)

    for d in dates:
        # optional business day filter
        if cfg.trading_days_only and not _is_business_day(d):
            continue

        desired_start = d.replace(
            hour=cfg.desired_start_hour, minute=0, second=0, microsecond=0
        )
        desired_end = d.replace(
            hour=cfg.desired_end_hour, minute=0, second=0, microsecond=0
        )

        # snap start to nearest bar
        start_ts = _nearest_index_time(idx, desired_start, tol)
        if start_ts is None:
            continue

        # primary attempt: the "next" bar after start for the 4h end
        # (many 4h grids are strictly every 4 hours)
        pos_after = idx.searchsorted(start_ts) + 1
        end_ts = None
        if pos_after < len(idx):
            candidate = idx[pos_after]
            # accept if between 2h and 6h after start (covers DST edges)
            delta_h = (candidate - start_ts).total_seconds() / 3600.0
            if 2.0 <= delta_h <= 6.0:
                end_ts = candidate

        # fallback: snap desired_end independently
        if end_ts is None:
            end_ts = _nearest_index_time(idx, desired_end, tol)

        if end_ts is None or end_ts <= start_ts:
            continue

        start_px = float(df_tkr.loc[start_ts, cfg.price_col])
        end_px = float(df_tkr.loc[end_ts, cfg.price_col])
        delta = end_px - start_px
        label = 1 if delta > 0.0 else 0

        out_rows.append(
            {
                "ts": start_ts,  # label time = the start bar time
                cfg.ticker_col: df_tkr[cfg.ticker_col].iloc[0],
                "label": label,
                "y": label,
                "Close_start": start_px,
                "Close_end": end_px,
                "delta": delta,
                "snapped_start": start_ts,
                "snapped_end": end_ts,
                "desired_start": desired_start,
                "desired_end": desired_end,
            }
        )

    if not out_rows:
        return pd.DataFrame(columns=[cfg.ticker_col, "label", "y", "Close_start", "Close_end", "delta"])

    out = pd.DataFrame(out_rows).set_index("ts").sort_index()
    return out[[cfg.ticker_col, "label", "y", "Close_start", "Close_end", "delta"]]


def make_window_label_auto(
    df: pd.DataFrame,
    start_hour: int,
    end_hour: int,
    *,
    ticker_col: str = "Ticker",
    price_col: str = "Close",
    trading_days_only: bool = True,
    duration_hours: int = 4,
    snap_tolerance_hours: float = 2.1,
) -> pd.DataFrame:
    """
    Auto-snaps the desired daily window (e.g., 09->13) to your sheet's 4h grid.
    Works per-ticker and tolerates different 4h anchors (…03,07,11,15… etc).
    """
    if df.empty:
        return pd.DataFrame(columns=[ticker_col, "label", "y", "Close_start", "Close_end", "delta"])

    for col in (ticker_col, price_col):
        if col not in df.columns:
            raise ValueError(f"labels.make_window_label_auto: missing column '{col}'")

    cfg = SnapConfig(
        desired_start_hour=start_hour,
        desired_end_hour=end_hour,
        duration_hours=duration_hours,
        snap_tolerance_hours=snap_tolerance_hours,
        trading_days_only=trading_days_only,
        ticker_col=ticker_col,
        price_col=price_col,
    )

    frames = []
    for tkr, g in df.groupby(ticker_col):
        frames.append(_labels_for_ticker(g.sort_index(), cfg))
    if not frames:
        return pd.DataFrame(columns=[ticker_col, "label", "y", "Close_start", "Close_end", "delta"])

    out = pd.concat(frames, axis=0).sort_index()
    return out


# Keep the old explicit-hour labeler (used earlier). Still available if needed.
def make_window_label(
    df: pd.DataFrame,
    start_hour: int,
    end_hour: int,
    threshold: float = 0.0,
    *,
    ticker_col: str = "Ticker",
    price_col: str = "Close",
    trading_days_only: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[ticker_col, "label", "y", "Close_start", "Close_end", "delta"])

    work = df[[ticker_col, price_col]].copy()
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[price_col])

    h = work.index.hour
    dfa = work[h == start_hour].copy()
    dfb = work[h == end_hour].copy()

    if trading_days_only:
        dfa = dfa[dfa.index.dayofweek < 5]
        dfb = dfb[dfb.index.dayofweek < 5]

    dfa["Date"] = dfa.index.normalize()
    dfb["Date"] = dfb.index.normalize()

    left = dfa.rename(columns={price_col: "Close_start"})
    right = dfb.rename(columns={price_col: "Close_end"})

    left_reset = left.reset_index()
    idx_name = left.index.name or left_reset.columns[0]
    left_reset = left_reset.rename(columns={idx_name: "ts"})

    right_reset = right.reset_index()[[ticker_col, "Date", "Close_end"]]

    merged = (
        left_reset
        .merge(right_reset, on=[ticker_col, "Date"], how="inner", validate="m:1")
        .set_index("ts")
        .sort_index()
    )
    merged["delta"] = merged["Close_end"] - merged["Close_start"]
    merged["label"] = (merged["delta"] > float(threshold)).astype("int8")
    merged["y"] = merged["label"].astype("int8")
    out = merged[[ticker_col, "label", "y", "Close_start", "Close_end", "delta"]].copy()
    out.index.name = None
    return out


# Back-compat alias
def make_8to9_label(df, *args, **kwargs):
    return make_window_label(df, 8, 9, *args, **kwargs)
