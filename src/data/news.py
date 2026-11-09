from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils import ensure_dirs, save_json

# Public RSS endpoints — content licensing remains with publishers.
RSS_FEEDS = [
    {"name": "dawn_business", "url": "https://www.dawn.com/feeds/business", "lang": "en"},
    {"name": "dawn_latest", "url": "https://www.dawn.com/feeds/home", "lang": "en"},
    {"name": "tribune_business", "url": "https://tribune.com.pk/feed/business", "lang": "en"},
    {"name": "brecorder", "url": "https://www.brecorder.com/feeds/latest-news", "lang": "en"},
]


def fetch_rss_headlines(max_per_feed: int = 40) -> pd.DataFrame:
    """Collect recent headlines only (not a historical archive)."""
    paths = ensure_dirs()
    try:
        import feedparser
    except ImportError as exc:
        raise ImportError("Install feedparser to fetch news RSS feeds") from exc

    rows = []
    errors = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:max_per_feed]:
                published = None
                if getattr(entry, "published_parsed", None):
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                rows.append(
                    {
                        "feed": feed["name"],
                        "title": getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", ""),
                        "link": getattr(entry, "link", ""),
                        "published_utc": published,
                        "lang": feed["lang"],
                        "collected_at": datetime.now(timezone.utc),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append({"feed": feed["name"], "error": str(exc)})

    df = pd.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = paths["data_raw"] / "news" / f"rss_{stamp}.csv"
    df.to_csv(out, index=False)
    save_json(
        {
            "feeds_attempted": len(RSS_FEEDS),
            "rows": len(df),
            "errors": errors,
            "limitation": (
                "RSS provides recent headlines only. Multi-year historical news text for NLP "
                "is not manufactured. Event-level historical sentiment remains null unless "
                "archived article text is supplied under data/raw/news/historical/."
            ),
        },
        paths["data_raw"] / "news" / "provenance.json",
    )
    return df


def load_historical_news_if_any() -> pd.DataFrame:
    paths = ensure_dirs()
    hist_dir = paths["data_raw"] / "news" / "historical"
    hist_dir.mkdir(parents=True, exist_ok=True)
    files = list(hist_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(
            columns=["date", "source", "title", "text", "url", "lang", "is_market_report"]
        )
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df
