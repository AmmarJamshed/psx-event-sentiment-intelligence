from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_metrics(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """Compute returns, volatility, drawdowns, volume deviations, abnormal sessions."""
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out["return_1d"] = out["Close"].pct_change()
    out["log_return_1d"] = np.log(out["Close"]).diff()
    out["return_1w"] = out["Close"].pct_change(5)
    out["return_1m"] = out["Close"].pct_change(21)

    out["vol_20d"] = out["return_1d"].rolling(20).std()
    out["vol_60d"] = out["return_1d"].rolling(60).std()
    out["vol_ann_20d"] = out["vol_20d"] * np.sqrt(245)

    rolling_mean = out["return_1d"].rolling(60).mean()
    rolling_std = out["return_1d"].rolling(60).std()
    out["abnormal_return"] = out["return_1d"] - rolling_mean
    out["abnormal_z"] = out["abnormal_return"] / rolling_std.replace(0, np.nan)
    out["is_abnormal"] = out["abnormal_z"].abs() >= z_threshold
    out["abnormal_direction"] = np.where(
        out["is_abnormal"],
        np.where(out["abnormal_z"] > 0, "positive", "negative"),
        "normal",
    )

    peak = out["Close"].cummax()
    out["drawdown"] = out["Close"] / peak - 1.0
    out["max_drawdown_252d"] = out["drawdown"].rolling(252, min_periods=20).min()

    if "Volume" in out.columns:
        vol = pd.to_numeric(out["Volume"], errors="coerce")
        out["volume"] = vol
        out["volume_ma20"] = vol.rolling(20).mean()
        out["volume_deviation"] = vol / out["volume_ma20"] - 1.0
    else:
        out["volume"] = np.nan
        out["volume_ma20"] = np.nan
        out["volume_deviation"] = np.nan

    # Market breadth proxy unavailable without advance/decline tape; leave null.
    out["advance_decline_ratio"] = np.nan
    out["market_breadth"] = np.nan

    out["prev_return"] = out["return_1d"].shift(1)
    out["trend_20d"] = out["Close"] / out["Close"].shift(20) - 1.0
    out["trend_60d"] = out["Close"] / out["Close"].shift(60) - 1.0
    return out


def largest_moves(metrics: pd.DataFrame, n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        empty = pd.DataFrame()
        return empty, empty
    abn = metrics.dropna(subset=["abnormal_z"]).copy()
    pos = abn.sort_values("abnormal_z", ascending=False).head(n)
    neg = abn.sort_values("abnormal_z", ascending=True).head(n)
    cols = [
        "Close",
        "return_1d",
        "abnormal_return",
        "abnormal_z",
        "vol_20d",
        "volume",
        "volume_deviation",
        "drawdown",
    ]
    cols = [c for c in cols if c in abn.columns]
    return pos[cols], neg[cols]
