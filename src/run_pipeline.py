from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Allow `python -m src.run_pipeline` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.events import export_events, load_events, seed_events_if_missing
from src.data.macro import collect_macro_data
from src.data.market import collect_market_data, compute_market_metrics, load_or_collect_market
from src.data.news import fetch_rss_headlines, load_historical_news_if_any
from src.event_study import map_events_to_market, rank_event_impacts
from src.features import build_feature_matrix
from src.models import fit_arimax, fit_ols_statsmodels, fit_sklearn_models, fit_var
from src.nlp.sentiment import build_daily_sentiment
from src.reports import build_dashboard, write_daily_intelligence, write_research_questions, write_research_report
from src.utils import ensure_dirs, save_json
from src.validation import crisis_vs_stable_performance, expanding_window_validation, residual_diagnostics, rolling_window_validation


def run_all(refresh_market: bool = True) -> dict:
    paths = ensure_dirs()
    limitations = []
    summary: dict = {"steps": {}}

    # 1) Events
    seed_events_if_missing()
    events = load_events()
    export_events(events)
    summary["steps"]["events"] = {"count": len(events)}

    # 2) Market
    try:
        market_raw = collect_market_data() if refresh_market else load_or_collect_market()
    except Exception as exc:  # noqa: BLE001
        limitations.append(f"Market download failed ({exc}); attempting local cache.")
        market_raw = load_or_collect_market()
    market = compute_market_metrics(market_raw)
    summary["steps"]["market"] = {
        "rows": len(market),
        "start": str(market.index.min().date()),
        "end": str(market.index.max().date()),
    }
    limitations.append(
        "Primary equity series is the PSX Analyzer archival KSE-100 close panel (local CSV / "
        "download/macro_kse100.csv; historically blended PSX portal + Yahoo). Official sector "
        "indices, advance/decline and full market-cap breadth are not included unless supplied."
    )

    # 3) Macro
    try:
        macro = collect_macro_data()
        summary["steps"]["macro"] = {"cols": list(macro.columns)}
    except Exception as exc:  # noqa: BLE001
        macro = None
        limitations.append(f"Macro download failed: {exc}")
        summary["steps"]["macro"] = {"status": "failed"}

    # 4) Event study
    mapped = map_events_to_market(events, market)
    rank_event_impacts(mapped, top_n=50)
    summary["steps"]["event_study"] = {"mapped": len(mapped)}
    limitations.append(
        "Event sentiment_score fields are null until historical article text is provided; "
        "severity blends curated priors with realised abnormal z-scores (partly ex-post)."
    )

    # 5) News + sentiment (recent RSS + optional historical CSVs)
    news_hist = load_historical_news_if_any()
    try:
        news_rss = fetch_rss_headlines()
    except Exception as exc:  # noqa: BLE001
        news_rss = None
        limitations.append(f"RSS fetch failed: {exc}")

    frames = []
    if news_hist is not None and not news_hist.empty:
        frames.append(news_hist)
    if news_rss is not None and not news_rss.empty:
        tmp = news_rss.copy()
        if "published_utc" in tmp.columns:
            tmp["date"] = tmp["published_utc"]
        if "summary" in tmp.columns and "text" not in tmp.columns:
            tmp["text"] = tmp["summary"]
        frames.append(tmp)
    news = pd_concat(frames)
    sentiment = build_daily_sentiment(news)
    if sentiment.empty or sentiment["avg_sentiment"].notna().sum() < 30:
        limitations.append(
            "Reliable multi-year historical sentiment is unavailable in this run. "
            "Recent RSS may populate only a short window; do not invent backfilled scores."
        )
    summary["steps"]["sentiment"] = {
        "news_rows": 0 if news is None else len(news),
        "sentiment_days": 0 if sentiment is None else int(sentiment["avg_sentiment"].notna().sum()) if "avg_sentiment" in sentiment.columns else 0,
    }

    # 6) Features + models
    features = build_feature_matrix(market, events, macro=macro, sentiment=sentiment)
    model_results = fit_sklearn_models(features, target="target_ret")
    # Predictive lead-1 target
    try:
        lead_results = fit_sklearn_models(features, target="target_ret_lead1")
    except Exception as exc:  # noqa: BLE001
        lead_results = None
        limitations.append(f"Lead-1 predictive models skipped: {exc}")

    ols_info = fit_ols_statsmodels(features, target="target_ret")
    fit_var(features)
    fit_arimax(features)

    # 7) Validation
    try:
        expanding_window_validation(features)
        rolling_window_validation(features)
        diagnostics = residual_diagnostics(features)
    except Exception as exc:  # noqa: BLE001
        diagnostics = {"error": str(exc)}
        limitations.append(f"Validation diagnostics partial failure: {exc}")
    crisis_vs_stable_performance(features, events)

    # 8) Reports / dashboard
    write_research_report(market, mapped, model_results, diagnostics, limitations)
    write_research_questions(mapped, market, features)
    write_daily_intelligence(news_rss if news_rss is not None else news, sentiment, market)
    dash = build_dashboard(market, mapped, sentiment)

    summary["limitations"] = limitations
    summary["ols"] = ols_info
    summary["dashboard"] = str(dash)
    summary["outputs"] = str(paths["output"])
    if lead_results is not None:
        summary["steps"]["lead1_models"] = "ok"
    save_json(summary, paths["output"] / "run_summary.json")
    return summary


def pd_concat(frames):
    import pandas as pd

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    parser = argparse.ArgumentParser(description="PSX Event & Sentiment Intelligence pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--no-refresh-market", action="store_true", help="Reuse cached market CSV if present")
    args = parser.parse_args()
    if not args.all:
        parser.print_help()
        print("\nTip: python -m src.run_pipeline --all")
        return
    try:
        summary = run_all(refresh_market=not args.no_refresh_market)
        print("Pipeline complete.")
        print(f"Outputs: {summary.get('outputs')}")
        print(f"Dashboard: {summary.get('dashboard')}")
        for lim in summary.get("limitations", []):
            print(f"LIMITATION: {lim}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
