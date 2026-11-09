from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORY_FLAGS = {
    "political": "political_instability",
    "economic": "economic_uncertainty",
    "geopolitical": "geopolitical_risk",
    "regulatory": "regulatory_announcement",
}


def build_feature_panel(
    market: pd.DataFrame,
    events_mapped: pd.DataFrame,
    sentiment_daily: pd.DataFrame | None = None,
    macros: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Consolidate market, event, sentiment, and macro features on trading days."""
    if market.empty:
        return pd.DataFrame()

    panel = market.copy()
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel = panel.sort_index()

    # Event indicators aggregated to trading day
    event_cols = [
        "event_any",
        "event_severity_max",
        "event_count",
        "political_instability",
        "economic_uncertainty",
        "geopolitical_risk",
        "regulatory_announcement",
        "election_period",
        "imf_announcement",
    ]
    for c in event_cols:
        panel[c] = 0.0

    if events_mapped is not None and not events_mapped.empty:
        em = events_mapped.copy()
        em["effective_trading_date"] = pd.to_datetime(em["effective_trading_date"])
        for dt, grp in em.groupby("effective_trading_date"):
            if dt not in panel.index:
                continue
            panel.loc[dt, "event_any"] = 1.0
            panel.loc[dt, "event_count"] = float(len(grp))
            panel.loc[dt, "event_severity_max"] = float(pd.to_numeric(grp["severity_prior"], errors="coerce").max())
            cats = set(grp["category"].astype(str).str.lower())
            for cat, flag in CATEGORY_FLAGS.items():
                if cat in cats:
                    panel.loc[dt, flag] = 1.0
            titles = " ".join(grp["title"].astype(str).tolist()).lower()
            subs = " ".join(grp.get("subcategory", pd.Series(dtype=str)).astype(str).tolist()).lower()
            blob = titles + " " + subs
            if "election" in blob:
                panel.loc[dt, "election_period"] = 1.0
            if "imf" in blob:
                panel.loc[dt, "imf_announcement"] = 1.0

    # Sentiment join
    sent_defaults = {
        "sentiment_avg": np.nan,
        "sentiment_neg_intensity": np.nan,
        "sentiment_pos_intensity": np.nan,
        "fear_index": np.nan,
        "uncertainty_index": np.nan,
        "news_volume": np.nan,
        "sentiment_momentum": np.nan,
        "sentiment_divergence": np.nan,
        "sentiment_coverage": 0.0,
    }
    for k, v in sent_defaults.items():
        panel[k] = v

    if sentiment_daily is not None and not sentiment_daily.empty:
        s = sentiment_daily.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        for col in sent_defaults:
            if col in s.columns:
                panel[col] = s[col].reindex(panel.index)
        panel["sentiment_coverage"] = panel["sentiment_avg"].notna().astype(float)

    # Macro joins (use lagged close returns where relevant)
    macros = macros or {}
    for name, df in macros.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        m = df.copy()
        m.index = pd.to_datetime(m.index).tz_localize(None)
        close = m["Close"].reindex(panel.index).ffill()
        panel[f"{name.lower()}_level"] = close
        panel[f"{name.lower()}_return"] = close.pct_change()

    # Targets: next-day return (no leakage — features at t predict t+1)
    panel["return_t"] = panel["return_1d"]
    panel["return_t1"] = panel["return_1d"].shift(-1)
    panel["direction_t1"] = (panel["return_t1"] > 0).astype(float)

    # Lagged sentiment to test precedence
    for lag in (1, 2, 3, 5):
        if "sentiment_avg" in panel.columns:
            panel[f"sentiment_avg_l{lag}"] = panel["sentiment_avg"].shift(lag)
        panel[f"event_severity_max_l{lag}"] = panel["event_severity_max"].shift(lag)

    return panel


FEATURE_COLUMNS = [
    "prev_return",
    "vol_20d",
    "vol_60d",
    "volume_deviation",
    "trend_20d",
    "trend_60d",
    "sentiment_avg",
    "sentiment_neg_intensity",
    "sentiment_pos_intensity",
    "fear_index",
    "uncertainty_index",
    "news_volume",
    "sentiment_momentum",
    "sentiment_divergence",
    "event_any",
    "event_severity_max",
    "event_count",
    "political_instability",
    "economic_uncertainty",
    "geopolitical_risk",
    "regulatory_announcement",
    "election_period",
    "imf_announcement",
    "usdpkr_return",
    "oil_return",
    "gold_return",
    "sp500_return",
    "em_return",
]


def select_model_matrix(panel: pd.DataFrame, target: str = "return_t1") -> tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    data = panel[cols + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    # For sentiment-sparse history, fill missing sentiment with 0 but keep coverage flag if present
    sent_cols = [c for c in cols if c.startswith("sentiment") or c in {"fear_index", "uncertainty_index", "news_volume"}]
    X = data[cols].copy()
    X[sent_cols] = X[sent_cols].fillna(0.0)
    X = X.fillna(0.0)
    y = data[target]
    return X, y
