from __future__ import annotations

import numpy as np
import pandas as pd


def _next_trading_day(idx: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    pos = idx.searchsorted(pd.Timestamp(dt).normalize())
    if pos >= len(idx):
        return pd.NaT
    return idx[pos]


def map_events_to_market(
    events: pd.DataFrame,
    market: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Map each event to trading-day reactions across multiple horizons.

    Uses announcement_date as the investor information date, then snaps to the
    next available trading session (handles weekends/holidays/after-close).
    """
    windows = windows or [0, 1, 3, 5, 10, 20]
    if events.empty or market.empty or "return_1d" not in market.columns:
        return pd.DataFrame()

    idx = market.index.sort_values()
    rows = []
    for _, ev in events.iterrows():
        info_date = ev.get("announcement_date")
        if pd.isna(info_date):
            info_date = ev.get("event_date")
        if pd.isna(info_date):
            continue

        # If explicitly after-close, shift information to next calendar day before snap.
        etime = str(ev.get("event_time", "unknown")).lower()
        snap_from = pd.Timestamp(info_date).normalize()
        if "after" in etime:
            snap_from = snap_from + pd.Timedelta(days=1)

        t0 = _next_trading_day(idx, snap_from)
        if pd.isna(t0):
            continue
        loc = idx.get_loc(t0)
        if isinstance(loc, slice):
            loc = loc.start

        row = {
            "event_id": ev["event_id"],
            "event_date": ev.get("event_date"),
            "announcement_date": ev.get("announcement_date"),
            "effective_trading_date": t0,
            "category": ev.get("category"),
            "subcategory": ev.get("subcategory"),
            "title": ev.get("title"),
            "source": ev.get("source"),
            "verification_status": ev.get("verification_status"),
            "severity_prior": ev.get("severity_prior"),
            "expected_direction_prior": ev.get("expected_direction_prior"),
            "event_time": ev.get("event_time"),
        }

        # T-1
        if loc - 1 >= 0:
            t_m1 = idx[loc - 1]
            row["return_tm1"] = float(market.loc[t_m1, "return_1d"]) if pd.notna(market.loc[t_m1, "return_1d"]) else np.nan
            row["close_tm1"] = float(market.loc[t_m1, "Close"])
        else:
            row["return_tm1"] = np.nan
            row["close_tm1"] = np.nan

        for w in windows:
            if loc + w >= len(idx):
                row[f"return_t{w}"] = np.nan
                row[f"cum_return_t{w}"] = np.nan
                continue
            tw = idx[loc + w]
            row[f"return_t{w}"] = float(market.loc[tw, "return_1d"]) if pd.notna(market.loc[tw, "return_1d"]) else np.nan
            # cumulative from T to T+w inclusive
            window_idx = idx[loc : loc + w + 1]
            rets = market.loc[window_idx, "return_1d"]
            row[f"cum_return_t{w}"] = float((1 + rets.fillna(0)).prod() - 1)

        if "volume" in market.columns:
            row["volume_t0"] = float(market.loc[t0, "volume"]) if pd.notna(market.loc[t0, "volume"]) else np.nan
            row["volume_deviation_t0"] = (
                float(market.loc[t0, "volume_deviation"])
                if "volume_deviation" in market.columns and pd.notna(market.loc[t0, "volume_deviation"])
                else np.nan
            )
        else:
            row["volume_t0"] = np.nan
            row["volume_deviation_t0"] = np.nan

        if "abnormal_z" in market.columns:
            row["abnormal_z_t0"] = float(market.loc[t0, "abnormal_z"]) if pd.notna(market.loc[t0, "abnormal_z"]) else np.nan
            row["abnormal_return_t0"] = (
                float(market.loc[t0, "abnormal_return"]) if pd.notna(market.loc[t0, "abnormal_return"]) else np.nan
            )
        else:
            row["abnormal_z_t0"] = np.nan
            row["abnormal_return_t0"] = np.nan

        # Observed market direction on event day
        r0 = row.get("return_t0")
        if r0 is None or (isinstance(r0, float) and np.isnan(r0)):
            row["market_direction"] = "unknown"
        elif r0 > 0.002:
            row["market_direction"] = "positive"
        elif r0 < -0.002:
            row["market_direction"] = "negative"
        else:
            row["market_direction"] = "neutral"

        # Impact score: severity-weighted absolute abnormal move if available
        az = row.get("abnormal_z_t0")
        sev = row.get("severity_prior")
        if az is not None and not (isinstance(az, float) and np.isnan(az)):
            row["impact_score"] = abs(float(az)) * (float(sev) / 10.0 if pd.notna(sev) else 1.0)
        elif r0 is not None and not (isinstance(r0, float) and np.isnan(r0)):
            row["impact_score"] = abs(float(r0)) * 100.0 * (float(sev) / 10.0 if pd.notna(sev) else 1.0)
        else:
            row["impact_score"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def rank_event_impacts(mapped: pd.DataFrame, n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapped.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = mapped.dropna(subset=["return_t0"]).copy()
    pos = df.sort_values("return_t0", ascending=False).head(n)
    neg = df.sort_values("return_t0", ascending=True).head(n)
    return pos, neg
