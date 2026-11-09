from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_impact_report(pos: pd.DataFrame, neg: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Event Impact Report — Top Market Reactions",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "Ranking uses **observed KSE-100 event-day return** after snapping announcement dates",
        "to the next trading session. This is association, not proven causation.",
        "",
        "## Largest positive event-day returns",
        "",
    ]
    if pos.empty:
        lines.append("_No mapped positive events (market data may be incomplete)._")
    else:
        lines.append("| Rank | Date | Return | Category | Title | Source |")
        lines.append("|---:|---|---:|---|---|---|")
        for i, (_, r) in enumerate(pos.iterrows(), 1):
            lines.append(
                f"| {i} | {pd.Timestamp(r['effective_trading_date']).date()} | "
                f"{r['return_t0']*100:.2f}% | {r.get('category','')} | {r.get('title','')} | {r.get('source','')} |"
            )

    lines += ["", "## Largest negative event-day returns", ""]
    if neg.empty:
        lines.append("_No mapped negative events (market data may be incomplete)._")
    else:
        lines.append("| Rank | Date | Return | Category | Title | Source |")
        lines.append("|---:|---|---:|---|---|---|")
        for i, (_, r) in enumerate(neg.iterrows(), 1):
            lines.append(
                f"| {i} | {pd.Timestamp(r['effective_trading_date']).date()} | "
                f"{r['return_t0']*100:.2f}% | {r.get('category','')} | {r.get('title','')} | {r.get('source','')} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_research_report(
    events_summary: dict,
    market_meta: dict,
    model_results: dict,
    research_answers: dict,
    path: Path,
) -> None:
    models = model_results.get("models", {})
    lines = [
        "# Statistical Research Report — PSX Event & Sentiment Intelligence",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## 1. Data coverage",
        "",
        f"- Events: **{events_summary.get('n_events')}** "
        f"({events_summary.get('date_min')} → {events_summary.get('date_max')})",
        f"- Categories: `{json.dumps(events_summary.get('by_category', {}))}`",
        f"- Market series: `{json.dumps(market_meta)}`",
        "",
        "## 2. Modelling protocol",
        "",
        "- Chronological train / validation / test split (no random shuffle).",
        "- Target: next-day KSE-100 return (`return_t1`).",
        "- Sentiment historical coverage is incomplete; missing values are not fabricated.",
        "- Correlation ≠ causation; event studies and Granger tests are associative.",
        "",
        f"- Split sizes: `{json.dumps(model_results.get('splits', {}))}`",
        "",
        "## 3. Model comparison (out-of-sample test)",
        "",
        "| Model | R² | RMSE | MAE | Directional Acc. | IC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, res in models.items():
        te = res.get("test", {})
        if not te:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {te.get('r2', float('nan')):.4f} | {te.get('rmse', float('nan')):.6f} | "
            f"{te.get('mae', float('nan')):.6f} | {te.get('directional_accuracy', float('nan')):.3f} | "
            f"{te.get('ic', float('nan')):.3f} |"
        )

    lines += ["", "## 4. OLS diagnostics", ""]
    ols = model_results.get("ols_diagnostics", {})
    if "summary_text" in ols:
        lines.append("```")
        lines.append(ols["summary_text"][:8000])
        lines.append("```")
        lines.append("")
        lines.append(f"- Condition number: `{ols.get('condition_number')}`")
        lines.append(f"- Durbin–Watson: `{ols.get('durbin_watson')}`")
        lines.append(f"- Breusch–Pagan p-value: `{ols.get('breusch_pagan_pvalue')}`")
    else:
        lines.append(f"`{json.dumps(ols)[:2000]}`")

    lines += ["", "## 5. Stationarity / Granger / VAR", ""]
    lines.append(f"- ADF: `{json.dumps(model_results.get('stationarity_adf', {}))}`")
    lines.append(f"- Granger (sentiment→returns): `{json.dumps(model_results.get('granger_sentiment_to_returns', {}))}`")
    lines.append(f"- VAR: `{json.dumps(model_results.get('var', {}))}`")

    lines += ["", "## 6. Research questions — evidence summary", ""]
    for q, a in research_answers.items():
        lines.append(f"### {q}")
        lines.append("")
        lines.append(str(a))
        lines.append("")

    lines += [
        "## 7. Limitations",
        "",
        "1. Historical public sentiment corpora are incomplete; do not treat filled zeros as observed neutrality.",
        "2. KSE-100 history from free vendors may have gaps (especially pre-2010 / post-Yahoo coverage breaks).",
        "3. Sector indices, advance/decline and full market-cap breadth often unavailable without licensed feeds.",
        "4. Event dates can differ from information arrival; we snap announcement dates to trading sessions.",
        "5. Predictive R² for daily equity returns is typically low; directional accuracy near 50% is common.",
        "6. This is research software, not investment advice.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_daily_intelligence(
    news: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
    latest_market: pd.Series | None,
    path: Path,
) -> None:
    lines = [
        "# Daily Intelligence Report — Pakistan / PSX",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Market snapshot",
        "",
    ]
    if latest_market is not None and len(latest_market):
        r = latest_market.get("return_1d", np.nan)
        lines.append(f"- Last close: **{latest_market.get('Close', float('nan')):,.2f}**")
        lines.append(f"- Last daily return: **{(r*100 if pd.notna(r) else float('nan')):.2f}%**")
        lines.append(f"- 20d ann. vol: **{latest_market.get('vol_ann_20d', float('nan'))}**")
    else:
        lines.append("_Market series unavailable in this run._")

    lines += ["", "## Public / news sentiment (contemporaneous only)", ""]
    if sentiment_daily is None or sentiment_daily.empty:
        lines.append("_No scored news in this run._")
        conf = "low"
        risk = "unknown"
    else:
        last = sentiment_daily.iloc[-1]
        lines.append(f"- Avg sentiment: **{last.get('sentiment_avg', float('nan')):.3f}** (−1 to +1)")
        lines.append(f"- Fear index: **{last.get('fear_index', float('nan')):.3f}**")
        lines.append(f"- Uncertainty: **{last.get('uncertainty_index', float('nan')):.3f}**")
        lines.append(f"- News volume scored: **{int(last.get('news_volume', 0))}**")
        s = float(last.get("sentiment_avg", 0) or 0)
        if s < -0.15:
            risk = "elevated downside skew (sentiment)"
        elif s > 0.15:
            risk = "constructive sentiment skew"
        else:
            risk = "balanced / mixed"
        conf = "medium" if int(last.get("news_volume", 0)) >= 8 else "low"

    lines += ["", "## Most reactionary headlines", ""]
    if news is None or news.empty:
        lines.append("_No headlines fetched._")
    else:
        for _, r in news.head(15).iterrows():
            lines.append(f"- [{r.get('source')}] {r.get('title')} ({r.get('link','')})")

    lines += [
        "",
        "## Potential PSX implications",
        "",
        f"- Expected risk direction (sentiment heuristic only): **{risk}**",
        f"- Confidence: **{conf}**",
        "- This is **not** a trading signal. Combine with macro, flows, and valuation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dashboard_html(
    market: pd.DataFrame,
    events_mapped: pd.DataFrame,
    sentiment_daily: pd.DataFrame | None,
    path: Path,
) -> None:
    """Lightweight self-contained HTML dashboard (Plotly CDN)."""
    m = market.dropna(subset=["Close"]).copy() if market is not None else pd.DataFrame()
    dates = [str(d.date()) for d in m.index] if not m.empty else []
    closes = [float(x) for x in m["Close"].tolist()] if not m.empty else []
    rets = [None if pd.isna(x) else float(x) for x in m.get("return_1d", pd.Series(dtype=float)).tolist()] if not m.empty else []

    ev_dates, ev_titles, ev_rets, ev_cats = [], [], [], []
    if events_mapped is not None and not events_mapped.empty:
        em = events_mapped.dropna(subset=["effective_trading_date"]).copy()
        for _, r in em.iterrows():
            ev_dates.append(str(pd.Timestamp(r["effective_trading_date"]).date()))
            ev_titles.append(str(r.get("title", "")))
            ev_rets.append(None if pd.isna(r.get("return_t0")) else float(r.get("return_t0")))
            ev_cats.append(str(r.get("category", "")))

    sent_dates, sent_vals, fear_vals = [], [], []
    if sentiment_daily is not None and not sentiment_daily.empty:
        for d, r in sentiment_daily.iterrows():
            sent_dates.append(str(pd.Timestamp(d).date()))
            sent_vals.append(None if pd.isna(r.get("sentiment_avg")) else float(r.get("sentiment_avg")))
            fear_vals.append(None if pd.isna(r.get("fear_index")) else float(r.get("fear_index")))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PSX Event & Sentiment Intelligence</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{ --bg:#0f1419; --card:#1a2332; --text:#e7ecf3; --muted:#8b9bb4; --accent:#3d9cf0; }}
  body {{ margin:0; font-family: "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:24px 32px; background:linear-gradient(120deg,#1a2332,#243044); border-bottom:1px solid #2a3a50; }}
  h1 {{ margin:0 0 6px; font-size:1.6rem; }}
  p {{ margin:0; color:var(--muted); }}
  main {{ display:grid; gap:16px; padding:16px 32px 40px; }}
  .row {{ display:grid; grid-template-columns:1fr; gap:16px; }}
  .card {{ background:var(--card); border-radius:12px; padding:8px; border:1px solid #2a3a50; }}
  .note {{ color:var(--muted); font-size:0.9rem; padding:0 8px 12px; }}
</style>
</head>
<body>
<header>
  <h1>PSX Event & Sentiment Intelligence</h1>
  <p>KSE-100 path, event markers, and contemporaneous sentiment — research dashboard</p>
</header>
<main>
  <div class="row"><div class="card"><div id="price"></div><div class="note">KSE-100 close with event markers (size ~ |event-day return|).</div></div></div>
  <div class="row"><div class="card"><div id="returns"></div></div></div>
  <div class="row"><div class="card"><div id="sentiment"></div><div class="note">Historical sentiment is shown only where news text was scored. Gaps are intentional.</div></div></div>
</main>
<script>
const dates = {json.dumps(dates)};
const closes = {json.dumps(closes)};
const rets = {json.dumps(rets)};
const evDates = {json.dumps(ev_dates)};
const evTitles = {json.dumps(ev_titles)};
const evRets = {json.dumps(ev_rets)};
const evCats = {json.dumps(ev_cats)};
const sentDates = {json.dumps(sent_dates)};
const sentVals = {json.dumps(sent_vals)};
const fearVals = {json.dumps(fear_vals)};

Plotly.newPlot('price', [
  {{x:dates, y:closes, type:'scatter', mode:'lines', name:'KSE-100', line:{{color:'#3d9cf0', width:1.5}}}},
  {{x:evDates, y:evDates.map((d)=>{{
      const i = dates.indexOf(d); return i>=0 ? closes[i] : null;
    }}), text:evTitles.map((t,i)=>t+' ('+evCats[i]+') ret='+((evRets[i]??0)*100).toFixed(2)+'%'),
    type:'scatter', mode:'markers', name:'Events',
    marker:{{size:evRets.map(r=>Math.min(18, 6+Math.abs(r||0)*120)), color:evRets.map(r=>(r||0)>=0?'#3ecf8e':'#f07178')}}}}
], {{paper_bgcolor:'#1a2332', plot_bgcolor:'#1a2332', font:{{color:'#e7ecf3'}},
    margin:{{t:40,r:20,b:40,l:50}}, title:'KSE-100 & Events'}});

Plotly.newPlot('returns', [
  {{x:dates, y:rets.map(v=>v===null?null:v*100), type:'bar', name:'Daily %', marker:{{color:rets.map(v=> (v||0)>=0?'#3ecf8e55':'#f0717855')}}}}
], {{paper_bgcolor:'#1a2332', plot_bgcolor:'#1a2332', font:{{color:'#e7ecf3'}},
    margin:{{t:40,r:20,b:40,l:50}}, title:'Daily returns (%)'}});

Plotly.newPlot('sentiment', [
  {{x:sentDates, y:sentVals, type:'scatter', mode:'lines+markers', name:'Sentiment', line:{{color:'#e0b15a'}}}},
  {{x:sentDates, y:fearVals, type:'scatter', mode:'lines', name:'Fear', line:{{color:'#f07178', dash:'dot'}}}}
], {{paper_bgcolor:'#1a2332', plot_bgcolor:'#1a2332', font:{{color:'#e7ecf3'}},
    margin:{{t:40,r:20,b:40,l:50}}, title:'Contemporaneous sentiment'}});
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def build_research_answers(
    events_mapped: pd.DataFrame,
    model_results: dict[str, Any],
    panel: pd.DataFrame,
) -> dict[str, str]:
    answers: dict[str, str] = {}
    if events_mapped is None or events_mapped.empty:
        base = "Insufficient mapped event–market observations in this run."
        for i in range(1, 13):
            answers[f"Q{i}"] = base
        return answers

    neg = events_mapped.nsmallest(10, "return_t0")
    pos = events_mapped.nlargest(10, "return_t0")
    answers["Q1 Which events linked to largest PSX declines?"] = (
        "Top negative event-day returns (associative):\n"
        + "\n".join(
            f"- {pd.Timestamp(r.effective_trading_date).date()} | {r.return_t0*100:.2f}% | {r.category} | {r.title}"
            for r in neg.itertuples()
        )
    )
    answers["Q2 Which events linked to largest PSX increases?"] = (
        "Top positive event-day returns (associative):\n"
        + "\n".join(
            f"- {pd.Timestamp(r.effective_trading_date).date()} | {r.return_t0*100:.2f}% | {r.category} | {r.title}"
            for r in pos.itertuples()
        )
    )

    def _avg(cat: str) -> float:
        sub = events_mapped.loc[events_mapped["category"].str.lower() == cat, "return_t0"]
        return float(sub.mean()) if len(sub) else float("nan")

    answers["Q3 Political instability vs economic uncertainty?"] = (
        f"Mean event-day return — political: {_avg('political')*100:.3f}%, "
        f"economic: {_avg('economic')*100:.3f}%. "
        "Compare absolute means and sample sizes before inferring strength; this is not a causal ranking."
    )

    cov = float(panel["sentiment_coverage"].mean()) if "sentiment_coverage" in panel.columns else 0.0
    answers["Q4 Does negative public sentiment precede declines?"] = (
        f"Sentiment coverage share of trading days in panel: {cov:.1%}. "
        f"Granger diagnostics: {json.dumps(model_results.get('granger_sentiment_to_returns', {}))}. "
        "If coverage is near zero historically, precedence cannot be claimed."
    )

    # Absorption speed: compare |cum returns| across windows
    windows = [1, 3, 5, 10, 20]
    abs_means = {}
    for w in windows:
        col = f"cum_return_t{w}"
        if col in events_mapped.columns:
            abs_means[w] = float(events_mapped[col].abs().mean())
    answers["Q5 How many days to absorb major news?"] = (
        f"Mean |cumulative return| by horizon: {abs_means}. "
        "If later horizons keep rising, adjustment may extend beyond day 0; if flat after T+1, absorption is fast."
    )

    imf = events_mapped[events_mapped["title"].astype(str).str.contains("IMF", case=False, na=False)]
    answers["Q6 Does positive IMF-related news improve confidence?"] = (
        f"IMF-tagged events n={len(imf)}; mean T return={(imf['return_t0'].mean()*100 if len(imf) else float('nan')):.3f}%, "
        f"mean T+5 cum={(imf['cum_return_t5'].mean()*100 if len(imf) and 'cum_return_t5' in imf else float('nan')):.3f}%."
    )

    if "usdpkr_return" in panel.columns and "return_t" in panel.columns:
        corr = float(panel[["usdpkr_return", "return_t"]].dropna().corr().iloc[0, 1])
    else:
        corr = float("nan")
    answers["Q7 Currency depreciation vs equity/sentiment?"] = (
        f"Corr(USD/PKR daily % change, KSE return) ≈ {corr:.3f} on overlapping days. "
        "Sentiment–FX links require contemporaneous text coverage."
    )
    answers["Q8 Social media beyond mainstream news?"] = (
        "Not evaluated in the default run (no licensed X/Reddit historical firehose). "
        "Pipeline supports adding social text as a separate channel for divergence features."
    )
    answers["Q9 Stronger sentiment predictability in crises?"] = (
        f"{json.dumps(model_results.get('regime_comparison', {}))}"
    )

    # Feature importance from best tree model if present
    fi = None
    for m in ("xgboost", "random_forest", "gradient_boosting"):
        if m in model_results.get("models", {}) and "feature_importance" in model_results["models"][m]:
            fi = model_results["models"][m]["feature_importance"]
            break
    if fi:
        top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:10]
        answers["Q10 Which sentiment indicators explain most?"] = f"Top importances: {top}"
    else:
        coefs = model_results.get("models", {}).get("ridge", {}).get("coefficients") or {}
        sent_coefs = {k: v for k, v in coefs.items() if "sentiment" in k or k in {"fear_index", "uncertainty_index"}}
        answers["Q10 Which sentiment indicators explain most?"] = f"Ridge sentiment-related coefficients: {sent_coefs}"

    contradict = events_mapped.copy()
    contradict["prior"] = contradict["expected_direction_prior"].astype(str).str.lower()
    contradict["obs"] = contradict["market_direction"].astype(str).str.lower()
    mask = (
        ((contradict["prior"] == "positive") & (contradict["obs"] == "negative"))
        | ((contradict["prior"] == "negative") & (contradict["obs"] == "positive"))
    )
    n_con = int(mask.sum())
    answers["Q11 Sentiment/prior vs market contradictions?"] = (
        f"{n_con} events where directional prior conflicted with event-day market direction "
        f"({n_con/max(len(events_mapped),1):.1%} of mapped events)."
    )
    answers["Q12 Can sentiment identify reversals?"] = (
        "Requires dense sentiment history around troughs/peaks. With sparse historical text, "
        "reversal detection is not validated out-of-sample in this build."
    )
    return answers
