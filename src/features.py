from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ensure_dirs


def build_feature_matrix(
    market: pd.DataFrame,
    events: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    paths = ensure_dirs()
    df = market[["close", "ret_1d", "vol_21d", "volume", "volume_z", "trend_63d", "abnormal_z"]].copy()
    df = df.rename(columns={"ret_1d": "psx_ret", "vol_21d": "hist_vol", "trend_63d": "prev_trend"})
    df["prev_ret"] = df["psx_ret"].shift(1)
    df["prev_vol"] = df["hist_vol"].shift(1)

    # Event dummies on investor info / effective trading dates
    ev = events.copy()
    ev["info_date"] = pd.to_datetime(ev["info_date"]).dt.normalize()
    cats = ["political", "economic", "geopolitical", "regulatory", "social"]
    for c in cats:
        mask_dates = set(ev.loc[ev["category"] == c, "info_date"])
        df[f"event_{c}"] = df.index.isin(mask_dates).astype(int)

    df["event_any"] = (df[[f"event_{c}" for c in cats]].sum(axis=1) > 0).astype(int)
    sev = ev.groupby("info_date")["severity_score"].max()
    df["event_severity"] = df.index.map(sev).fillna(0.0)
    df["election_period"] = df["event_political"]  # refined below
    df["imf_announcement"] = 0
    df["regulatory_announcement"] = df["event_regulatory"]
    if "subcategory" in ev.columns:
        imf_dates = set(ev.loc[ev["subcategory"] == "imf", "info_date"])
        elec_dates = set(ev.loc[ev["subcategory"] == "election", "info_date"])
        df["imf_announcement"] = df.index.isin(imf_dates).astype(int)
        df["election_period"] = df.index.isin(elec_dates).astype(int)

    df["political_instability"] = df["event_political"]
    df["geopolitical_risk"] = df["event_geopolitical"]
    df["economic_uncertainty"] = ((df["event_economic"] + df["imf_announcement"]) > 0).astype(int)

    if macro is not None and not macro.empty:
        m = macro.copy()
        m.index = pd.to_datetime(m.index).normalize()
        for col in ["usdpkr_ret", "oil_ret", "gold_ret", "sp500_ret", "emerging_ret"]:
            if col in m.columns:
                df[col] = m[col].reindex(df.index)
            else:
                # try derive from level columns
                base = col.replace("_ret", "")
                if base in m.columns:
                    df[col] = m[base].pct_change().reindex(df.index)
                else:
                    df[col] = np.nan
        for col in ["policy_rate", "inflation_cpi", "fx_reserves"]:
            if col in m.columns:
                df[col] = m[col].reindex(df.index)

    # Sentiment — only where observed; do not impute historical fiction
    sent_cols = [
        "avg_sentiment",
        "neg_intensity",
        "pos_intensity",
        "fear_index",
        "uncertainty_index",
        "news_volume",
        "sentiment_momentum",
        "sentiment_disagreement",
        "public_sentiment",
    ]
    for c in sent_cols:
        df[c] = np.nan
    if sentiment is not None and not sentiment.empty:
        s = sentiment.copy()
        s["date"] = pd.to_datetime(s["date"]).dt.normalize()
        s = s.set_index("date")
        for c in sent_cols:
            if c in s.columns:
                df[c] = s[c].reindex(df.index)

    # Lags for predictive specs (t features → t+1 return)
    for c in ["avg_sentiment", "neg_intensity", "event_severity", "news_volume", "public_sentiment"]:
        df[f"{c}_lag1"] = df[c].shift(1)

    df["target_ret"] = df["psx_ret"]
    df["target_ret_lead1"] = df["psx_ret"].shift(-1)

    out = paths["data_processed"] / "feature_matrix.csv"
    df.to_csv(out)
    df.to_csv(paths["output"] / "models" / "feature_matrix.csv")
    return df
