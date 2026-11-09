from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils import ensure_dirs, load_config

# Domain lexicon — transparent, inspectable fallback when transformers are disabled.
# Scores are heuristic; not a substitute for labelled Urdu/English financial corpora.
POS_TERMS = {
    "optimism": ["optimism", "optimistic", "hope", "recovery", "rebound", "surge", "rally", "boom"],
    "confidence": ["confidence", "stable", "stability", "reform", "approval", "agreement", "deal", "inflow"],
    "enthusiasm": ["enthusiasm", "bullish", "record high", "breakthrough", "boost", "upbeat"],
}
NEG_TERMS = {
    "fear": ["fear", "fears", "afraid", "scare", "threat"],
    "panic": ["panic", "crash", "collapse", "selloff", "sell-off", "rout"],
    "uncertainty": ["uncertainty", "uncertain", "unclear", "ambiguity", "confusion"],
    "anger": ["anger", "outrage", "furious", "protest", "clash"],
    "political_dissatisfaction": ["no-confidence", "dissolution", "martial", "emergency", "rigging", "crackdown"],
    "economic_anxiety": ["inflation", "devaluation", "default", "reserves", "shortage", "austerity", "imf pressure"],
    "investor_pessimism": ["bearish", "outflow", "downgrade", "loss", "decline", "slump", "plunge"],
}

MARKET_REPORT_PATTERNS = [
    r"\bkse[- ]?100\b",
    r"\bpsx\b",
    r"\bindex (fell|rose|closed|gained|lost)\b",
    r"\bpoints\b",
    r"\bmarket (closed|ended|finished)\b",
]


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\u0600-\u06ff\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_market_performance_report(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in MARKET_REPORT_PATTERNS)


def _count_hits(text: str, terms: Iterable[str]) -> int:
    return sum(1 for term in terms if term in text)


def lexicon_score(text: str) -> dict:
    cleaned = _clean(text)
    if not cleaned:
        return {
            "sentiment_score": np.nan,
            "is_market_report": False,
            "method": "lexicon",
            **{k: 0.0 for k in list(POS_TERMS) + list(NEG_TERMS)},
        }

    pos_dims = {k: _count_hits(cleaned, v) for k, v in POS_TERMS.items()}
    neg_dims = {k: _count_hits(cleaned, v) for k, v in NEG_TERMS.items()}
    pos_raw = sum(pos_dims.values())
    neg_raw = sum(neg_dims.values())
    total = pos_raw + neg_raw
    if total == 0:
        score = 0.0
    else:
        score = (pos_raw - neg_raw) / total

    market_report = is_market_performance_report(cleaned)
    # Down-weight pure market-result reporting when estimating *public reaction* sentiment
    public_score = score * (0.35 if market_report else 1.0)

    out = {
        "sentiment_score": float(np.clip(public_score, -1, 1)),
        "raw_sentiment": float(np.clip(score, -1, 1)),
        "positive_intensity": float(pos_raw / max(total, 1)),
        "negative_intensity": float(neg_raw / max(total, 1)),
        "fear_optimism_ratio": float((neg_dims["fear"] + 1) / (pos_dims["optimism"] + 1)),
        "is_market_report": market_report,
        "method": "lexicon",
    }
    for k, v in {**pos_dims, **neg_dims}.items():
        out[k] = float(v)
    return out


def score_texts(texts: pd.Series | list[str]) -> pd.DataFrame:
    cfg = load_config()
    use_tf = bool(cfg["sentiment"].get("use_transformers"))
    rows = []
    if use_tf:
        try:
            rows = _transformer_scores(list(texts))
        except Exception:
            use_tf = False
    if not use_tf:
        rows = [lexicon_score(t if isinstance(t, str) else "") for t in texts]
    return pd.DataFrame(rows)


def _transformer_scores(texts: list[str]) -> list[dict]:
    """Optional FinBERT / multilingual path — only when explicitly enabled and installed."""
    from transformers import pipeline

    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    rows = []
    for text in texts:
        cleaned = _clean(text)[:512]
        if not cleaned:
            rows.append(lexicon_score(""))
            continue
        pred = clf(cleaned)[0]
        label = pred["label"].lower()
        conf = float(pred["score"])
        if label == "positive":
            score = conf
        elif label == "negative":
            score = -conf
        else:
            score = 0.0
        base = lexicon_score(cleaned)
        base["sentiment_score"] = float(np.clip(score, -1, 1))
        base["method"] = "finbert+lexicon_dims"
        rows.append(base)
    return rows


def build_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level scores to daily public vs market-news sentiment."""
    paths = ensure_dirs()
    if news_df is None or news_df.empty:
        empty = pd.DataFrame(
            columns=[
                "date",
                "avg_sentiment",
                "neg_intensity",
                "pos_intensity",
                "fear_index",
                "uncertainty_index",
                "news_volume",
                "sentiment_volatility",
                "fear_optimism_ratio",
                "sentiment_disagreement",
                "public_sentiment",
                "market_news_sentiment",
            ]
        )
        empty.to_csv(paths["output"] / "sentiment" / "daily_sentiment.csv", index=False)
        return empty

    df = news_df.copy()
    text_col = "text" if "text" in df.columns else "title"
    if "summary" in df.columns:
        df["_doc"] = df[text_col].fillna("") + " " + df["summary"].fillna("")
    else:
        df["_doc"] = df[text_col].fillna("")

    scores = score_texts(df["_doc"])
    scored = pd.concat([df.reset_index(drop=True), scores], axis=1)

    if "published_utc" in scored.columns:
        scored["date"] = pd.to_datetime(scored["published_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    elif "date" in scored.columns:
        scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    else:
        raise ValueError("News frame needs published_utc or date")

    public = scored.loc[~scored["is_market_report"].astype(bool)]
    marketish = scored.loc[scored["is_market_report"].astype(bool)]

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "avg_sentiment": g["sentiment_score"].mean(),
                "neg_intensity": g["negative_intensity"].mean(),
                "pos_intensity": g["positive_intensity"].mean(),
                "fear_index": g["fear"].mean() if "fear" in g else np.nan,
                "uncertainty_index": g["uncertainty"].mean() if "uncertainty" in g else np.nan,
                "news_volume": len(g),
                "sentiment_volatility": g["sentiment_score"].std(),
                "fear_optimism_ratio": g["fear_optimism_ratio"].mean(),
                "sentiment_disagreement": g["sentiment_score"].std(),
            }
        )

    daily = scored.groupby("date", group_keys=False).apply(_agg, include_groups=False).reset_index()
    pub = public.groupby("date")["sentiment_score"].mean().rename("public_sentiment")
    mkt = marketish.groupby("date")["sentiment_score"].mean().rename("market_news_sentiment")
    daily = daily.merge(pub, on="date", how="left").merge(mkt, on="date", how="left")
    daily["sentiment_momentum"] = daily["avg_sentiment"].diff()
    daily.to_csv(paths["output"] / "sentiment" / "daily_sentiment.csv", index=False)
    scored.to_csv(paths["output"] / "sentiment" / "article_scores.csv", index=False)
    return daily
