from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ensure_dirs, load_config, next_trading_day, previous_trading_day


def _cum_return(rets: pd.Series, start_idx: int, horizon: int) -> float:
    if start_idx < 0 or start_idx >= len(rets):
        return np.nan
    end = min(start_idx + horizon, len(rets) - 1)
    if end < start_idx:
        return np.nan
    window = rets.iloc[start_idx : end + 1]
    if window.isna().any():
        return np.nan
    return float((1 + window).prod() - 1)


def map_events_to_market(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Map each event to investor info date and multi-horizon market reactions."""
    cfg = load_config()
    paths = ensure_dirs()
    mkt = market.copy().sort_index()
    dates = mkt.index
    rets = mkt["ret_1d"]
    windows = cfg["market"]["windows"]

    rows = []
    for _, ev in events.iterrows():
        info_date = pd.Timestamp(ev["info_date"]).normalize()
        event_date = pd.Timestamp(ev["event_date"]).normalize()
        t0 = next_trading_day(dates, info_date)
        if t0 is None:
            continue
        t0_pos = dates.get_loc(t0)
        if isinstance(t0_pos, slice):
            t0_pos = t0_pos.start
        pre = previous_trading_day(dates, t0)

        row = {
            "event_id": ev["event_id"],
            "event_date": event_date,
            "info_date": info_date,
            "effective_trading_date": t0,
            "category": ev["category"],
            "subcategory": ev.get("subcategory"),
            "description": ev["description"],
            "source": ev["source"],
            "source_url": ev.get("source_url"),
            "severity_score": ev.get("severity_score"),
            "verified": ev.get("verified"),
            "close_t": float(mkt.loc[t0, "close"]),
            "volume_t": float(mkt.loc[t0, "volume"]) if "volume" in mkt.columns and pd.notna(mkt.loc[t0, "volume"]) else np.nan,
            "ret_t": float(rets.loc[t0]) if pd.notna(rets.loc[t0]) else np.nan,
            "abnormal_z_t": float(mkt.loc[t0, "abnormal_z"]) if "abnormal_z" in mkt.columns else np.nan,
            "ret_tm1": float(rets.loc[pre]) if pre is not None and pd.notna(rets.loc[pre]) else np.nan,
        }

        for h in windows:
            if h == 0:
                row["car_t0"] = row["ret_t"]
            else:
                row[f"car_t{h}"] = _cum_return(rets, t0_pos, h)

        # Immediate direction on effective trading day
        r = row["ret_t"]
        if pd.isna(r):
            direction = "unknown"
        elif r > 0.002:
            direction = "positive"
        elif r < -0.002:
            direction = "negative"
        else:
            direction = "neutral"
        row["market_direction_t"] = direction

        # Delayed reaction: sign of CAR to T+5 vs T
        car5 = row.get("car_t5")
        if pd.notna(car5) and pd.notna(r):
            if abs(car5) < 0.005:
                row["market_reaction"] = "muted"
            elif (car5 > 0) == (r > 0):
                row["market_reaction"] = "immediate_persistent"
            else:
                row["market_reaction"] = "reversal_within_5d"
        else:
            row["market_reaction"] = "insufficient_data"

        # Sentiment placeholder — filled only if NLP scores exist
        row["sentiment_score"] = np.nan
        row["sentiment_source"] = "unavailable_historical"
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        # Event severity blends curated prior with absolute abnormal z
        out["event_severity"] = (
            0.6 * out["severity_score"].astype(float).fillna(0.5)
            + 0.4 * (out["abnormal_z_t"].abs() / 5.0).clip(0, 1).fillna(0)
        )
        out = out.sort_values("event_date").reset_index(drop=True)

    out_path = paths["output"] / "event_study" / "event_market_map.csv"
    out.to_csv(out_path, index=False)
    return out


def rank_event_impacts(mapped: pd.DataFrame, top_n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_dirs()
    df = mapped.dropna(subset=["car_t5"]).copy()
    df["impact_score"] = df["car_t5"].abs() * (0.5 + 0.5 * df["event_severity"].fillna(0.5))

    neg = df.nsmallest(min(top_n, len(df)), "car_t5")
    pos = df.nlargest(min(top_n, len(df)), "car_t5")
    neg.to_csv(paths["output"] / "reports" / "top_negative_events.csv", index=False)
    pos.to_csv(paths["output"] / "reports" / "top_positive_events.csv", index=False)

    combined = pd.concat(
        [
            pos.assign(rank_type="positive_impact"),
            neg.assign(rank_type="negative_impact"),
        ],
        ignore_index=True,
    )
    combined.to_csv(paths["output"] / "reports" / "event_impact_top50.csv", index=False)
    return pos, neg
