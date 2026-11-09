from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fetch_rss_headlines(feeds: list[dict]) -> pd.DataFrame:
    """Fetch current RSS headlines for daily intelligence (not historical backfill)."""
    try:
        import feedparser
    except ImportError:
        return pd.DataFrame()

    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for feed in feeds:
        url = feed.get("url")
        name = feed.get("name", "unknown")
        if not url:
            continue
        try:
            parsed = feedparser.parse(url)
        except Exception:
            continue
        for e in parsed.entries[:40]:
            published = None
            if getattr(e, "published_parsed", None):
                published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            rows.append(
                {
                    "source": name,
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "link": getattr(e, "link", ""),
                    "published": published,
                    "fetched_at": fetched_at,
                }
            )
    return pd.DataFrame(rows)


def filter_pakistan_finance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [
        "pakistan",
        "psx",
        "kse",
        "imf",
        "rupee",
        "pkr",
        "sbp",
        "secp",
        "karachi",
        "islamabad",
        "lahore",
        "fatf",
        "budget",
        "inflation",
        "election",
        "army",
        "protest",
    ]
    blob = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    mask = blob.apply(lambda t: any(k in t for k in keys))
    return df.loc[mask].copy()


def build_daily_sentiment_from_news(news: pd.DataFrame) -> pd.DataFrame:
    from .lexicon import score_texts

    if news.empty:
        return pd.DataFrame()

    news = news.copy()
    if "published" in news.columns:
        news["date"] = pd.to_datetime(news["published"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    else:
        news["date"] = pd.Timestamp.utcnow().normalize()

    rows = []
    for dt, grp in news.groupby("date"):
        texts = (grp["title"].fillna("") + ". " + grp["summary"].fillna("")).tolist()
        scored = score_texts(texts)
        rows.append(
            {
                "date": dt,
                "sentiment_avg": scored.score,
                "sentiment_neg_intensity": scored.neg_intensity,
                "sentiment_pos_intensity": scored.pos_intensity,
                "fear_index": scored.emotions.get("fear", 0.0) + scored.emotions.get("panic", 0.0),
                "uncertainty_index": scored.emotions.get("uncertainty", 0.0),
                "news_volume": scored.news_volume,
                "sentiment_divergence": scored.disagreement,
                "sentiment_momentum": float("nan"),
                "notes": scored.notes,
            }
        )
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    if not daily.empty:
        daily["sentiment_momentum"] = daily["sentiment_avg"].diff()
    return daily


def save_news_snapshot(news: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    news.to_json(path, orient="records", indent=2, date_format="iso")


def write_sentiment_limitations(path: str | Path) -> None:
    text = """# Sentiment Data Limitations

Historical day-by-day public sentiment for Pakistan (English, Urdu, Roman Urdu)
from 2000 onward is **not reliably reconstructable** from free sources without
licensed news archives and social-firehose access.

This project therefore:

1. Scores **contemporaneous** RSS / collected articles with a transparent lexicon
   (optional transformer hook).
2. Leaves historical sentiment features as **missing** rather than fabricating
   scores for past events.
3. Distinguishes market-move reporting language from reactionary public sentiment.
4. Treats any historical event→sentiment cells without primary text as unavailable.

Do not interpret zeros filled at model training time as observed neutral sentiment;
see `sentiment_coverage` in the feature panel.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
