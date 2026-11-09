from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ROOT, ensure_dirs, load_config


def write_research_report(
    market: pd.DataFrame,
    mapped: pd.DataFrame,
    model_results: pd.DataFrame,
    diagnostics: dict,
    limitations: list[str],
) -> Path:
    paths = ensure_dirs()
    out = paths["output"] / "reports" / "statistical_research_report.md"

    test = model_results.loc[model_results["split"] == "test"].copy() if "split" in model_results.columns else model_results
    best = None
    if not test.empty and "rmse" in test.columns:
        best = test.sort_values("rmse").iloc[0]

    lines = [
        "# Statistical Research Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "This report summarises explanatory and out-of-sample predictive analysis of PSX returns",
        "conditioned on event indicators, macro controls, and (where available) sentiment features.",
        "",
        "## Market coverage",
        "",
        f"- Observations: **{len(market):,}**",
        f"- Date range: **{market.index.min().date()} → {market.index.max().date()}**",
        f"- Abnormal sessions (|z|≥threshold): **{int(market['abnormal_session'].sum()) if 'abnormal_session' in market else 'n/a'}**",
        "",
        "## Event study coverage",
        "",
        f"- Mapped events: **{len(mapped)}**",
        f"- Categories: {', '.join(sorted(mapped['category'].dropna().unique())) if len(mapped) else 'n/a'}",
        "",
        "## Model comparison (chronological holdout)",
        "",
    ]
    if not test.empty:
        show = test.copy()
        cols = [c for c in ["model", "r2", "rmse", "mae", "directional_accuracy", "information_coefficient", "n_features"] if c in show.columns]
        lines.append(show[cols].to_markdown(index=False))
    else:
        lines.append("_No model results available._")
    lines += ["", "## Best holdout model", ""]
    if best is not None:
        lines.append(
            f"**{best['model']}** — RMSE={best.get('rmse', float('nan')):.6f}, "
            f"R²={best.get('r2', float('nan')):.4f}, "
            f"directional accuracy={best.get('directional_accuracy', float('nan')):.3f}, "
            f"IC={best.get('information_coefficient', float('nan')):.3f}."
        )
    else:
        lines.append("_Unavailable._")

    lines += ["", "## OLS / residual diagnostics", "", "```json", str(diagnostics), "```", ""]
    lines += [
        "## Interpretation caveats",
        "",
        "- Coefficients and importances describe **associations**, not proven causation.",
        "- Event studies measure market reaction around investor information dates; competing news may confound attribution.",
        "- Sentiment features are sparse historically unless archived text is supplied.",
        "",
        "## Data & methodology limitations",
        "",
    ]
    for lim in limitations:
        lines.append(f"- {lim}")
    lines += [
        "",
        "## Files",
        "",
        "- `output/models/ols_coefficients_target_ret.csv`",
        "- `output/models/feature_importance_target_ret.json`",
        "- `output/models/model_comparison_target_ret.csv`",
        "- `output/event_study/event_market_map.csv`",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_research_questions(mapped: pd.DataFrame, market: pd.DataFrame, features: pd.DataFrame) -> Path:
    paths = ensure_dirs()
    out = paths["output"] / "reports" / "research_questions.md"

    m = mapped.dropna(subset=["car_t5"]) if "car_t5" in mapped.columns else mapped
    top_neg = m.nsmallest(10, "car_t5") if "car_t5" in m.columns and len(m) else pd.DataFrame()
    top_pos = m.nlargest(10, "car_t5") if "car_t5" in m.columns and len(m) else pd.DataFrame()

    pol = m.loc[m["category"] == "political", "car_t5"].abs().mean() if len(m) else np.nan
    eco = m.loc[m["category"] == "economic", "car_t5"].abs().mean() if len(m) else np.nan

    # Absorption: compare |CAR| growth across horizons
    horizons = [c for c in ["car_t0", "car_t1", "car_t3", "car_t5", "car_t10", "car_t20"] if c in m.columns]
    absorption = {h: float(m[h].abs().mean()) for h in horizons} if len(m) else {}

    imf_mask = pd.Series(False, index=m.index)
    if "subcategory" in m.columns:
        imf_mask |= m["subcategory"].astype(str).str.lower().str.contains("imf", na=False)
    for col in ("description", "title"):
        if col in m.columns:
            imf_mask |= m[col].astype(str).str.contains("IMF", case=False, na=False)
    imf = m.loc[imf_mask]
    imf_mean = float(imf["car_t5"].mean()) if len(imf) and "car_t5" in imf else float("nan")

    # Sentiment preceding declines — only if sentiment exists
    sent_avail = features["avg_sentiment"].notna().sum() if "avg_sentiment" in features.columns else 0
    if sent_avail > 50:
        tmp = features[["avg_sentiment", "psx_ret"]].dropna()
        tmp["decl"] = tmp["psx_ret"] < 0
        lag = tmp["avg_sentiment"].shift(1)
        neg_before = float(lag[tmp["decl"]].mean())
        pos_before = float(lag[~tmp["decl"]].mean())
        sent_answer = (
            f"Observed mean lag-1 sentiment before down days={neg_before:.3f} vs up days={pos_before:.3f} "
            f"(n={len(tmp)} overlapping days). This is associative, not causal."
        )
    else:
        sent_answer = (
            f"Historical daily sentiment coverage is insufficient (non-null days={sent_avail}). "
            "Cannot claim that negative public sentiment precedes declines without archived text."
        )

    fx_corr = float("nan")
    if "usdpkr_ret" in features.columns:
        sub = features[["psx_ret", "usdpkr_ret"]].dropna()
        if len(sub) > 50:
            fx_corr = float(sub.corr().iloc[0, 1])

    def _table(frame: pd.DataFrame) -> str:
        cols = [c for c in ["event_id", "event_date", "category", "description", "car_t5"] if c in frame.columns]
        if frame.empty or not cols:
            return "_Insufficient mapped events._"
        try:
            return frame[cols].to_markdown(index=False)
        except Exception:
            return frame[cols].to_string(index=False)

    pol_eco = (
        f"Mean |CAR T+5|: political={pol:.4f}, economic={eco:.4f}."
        if pd.notna(pol) and pd.notna(eco)
        else "Insufficient category coverage."
    )
    body = "\n".join(
        [
            "# Research Questions — Evidence Summary",
            "",
            "## 1. Largest PSX declines around events (by CAR T→T+5)",
            "",
            _table(top_neg),
            "",
            "## 2. Largest PSX increases around events (by CAR T→T+5)",
            "",
            _table(top_pos),
            "",
            "## 3. Political instability vs economic uncertainty",
            "",
            pol_eco,
            "",
            "## 4. Does negative public sentiment precede declines?",
            "",
            sent_answer,
            "",
            "## 5. How many days to absorb major news?",
            "",
            f"Mean absolute CAR by horizon: {absorption}",
            "",
            "## 6. Positive IMF-related news and confidence",
            "",
            f"Mean CAR T+5 on IMF-tagged events: {imf_mean:.4f} (n={len(imf)}).",
            "",
            "## 7. Currency depreciation vs equity returns",
            "",
            f"Corr(PSX return, USD/PKR return) ≈ {fx_corr:.3f} (positive USD/PKR return = PKR weaker).",
            "",
            "## 8. Social media beyond mainstream news",
            "",
            "Not evaluated in v0.1 — requires authenticated historical social firehose. Marked as future work.",
            "",
            "## 9. Stronger sentiment predictability in crises",
            "",
            "See `output/models/crisis_vs_stable.csv`. Sentiment interaction tests require denser sentiment history.",
            "",
            "## 10. Greatest explanatory sentiment indicators",
            "",
            "See `output/models/feature_importance_*.json` when sentiment columns survive missingness filters.",
            "",
            "## 11. Sentiment–market contradictions",
            "",
            "Inspect days where `public_sentiment` and `psx_ret` have opposite signs in `feature_matrix.csv` once sentiment exists.",
            "",
            "## 12. Sentiment as reversal indicator",
            "",
            "Requires contrarian tests on extreme sentiment quintiles with lead returns — deferred until sentiment panel is populated.",
            "",
        ]
    )
    out.write_text(body, encoding="utf-8")
    return out


def write_daily_intelligence(news: pd.DataFrame, sentiment: pd.DataFrame, market: pd.DataFrame) -> Path:
    paths = ensure_dirs()
    out = paths["output"] / "reports" / "daily_intelligence.md"
    last_close = market["close"].iloc[-1] if len(market) else float("nan")
    last_ret = market["ret_1d"].iloc[-1] if "ret_1d" in market.columns and len(market) else float("nan")
    last_date = market.index.max().date() if len(market) else "n/a"

    top_news = []
    if news is not None and not news.empty:
        cols = [c for c in ["title", "feed", "link"] if c in news.columns]
        top_news = news.head(8)[cols].fillna("").to_dict(orient="records")

    sent = float("nan")
    if sentiment is not None and not sentiment.empty and "avg_sentiment" in sentiment.columns:
        sent = float(sentiment["avg_sentiment"].iloc[-1])

    if pd.isna(sent):
        risk = "unknown"
        conf = "low"
        sent_txt = "n/a (insufficient scored text today)"
    elif sent < -0.15:
        risk = "elevated downside / risk-off bias"
        conf = "medium"
        sent_txt = f"{sent:.3f}"
    elif sent > 0.15:
        risk = "mild risk-on bias"
        conf = "medium"
        sent_txt = f"{sent:.3f}"
    else:
        risk = "balanced / inconclusive"
        conf = "low-medium"
        sent_txt = f"{sent:.3f}"

    lines = [
        "# Daily Intelligence Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Market snapshot",
        "",
        f"- Last session in dataset: **{last_date}**",
        f"- KSE-100 proxy close: **{last_close:,.2f}**",
        f"- Daily return: **{last_ret:.2%}**" if pd.notna(last_ret) else "- Daily return: n/a",
        "",
        "## Public / news sentiment",
        "",
        f"- Aggregate sentiment: **{sent_txt}**",
        f"- Expected risk direction: **{risk}**",
        f"- Confidence: **{conf}**",
        "",
        "## Most reactionary / recent headlines",
        "",
    ]
    if top_news:
        for n in top_news:
            lines.append(f"- {n.get('title', '')} _(source: {n.get('feed', '')})_")
            if n.get("link"):
                lines.append(f"  - {n['link']}")
    else:
        lines.append("- No RSS headlines collected in this run.")
    lines += [
        "",
        "## Potential PSX implications",
        "",
        "- Treat headline sentiment as a **soft prior**, not a trade signal.",
        "- Cross-check against FX (USD/PKR), oil, and scheduled SBP/IMF/political calendar events.",
        "- Same-day market-report articles are down-weighted so they do not masquerade as leading public sentiment.",
        "",
        "## Disclaimer",
        "",
        "Research prototype only. Not investment advice.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_dashboard(market: pd.DataFrame, mapped: pd.DataFrame, sentiment: pd.DataFrame) -> Path:
    paths = ensure_dirs()
    out = paths["output"] / "dashboard" / "index.html"

    mkt = market.tail(1500).copy()
    mkt_dates = [d.strftime("%Y-%m-%d") for d in mkt.index]
    closes = [None if pd.isna(x) else round(float(x), 2) for x in mkt["close"]]
    vols = [None if pd.isna(x) else round(float(x), 4) for x in mkt["ret_1d"].rolling(21).std()] if "ret_1d" in mkt else []

    ev_points = []
    if mapped is not None and not mapped.empty:
        for _, r in mapped.iterrows():
            if pd.isna(r.get("effective_trading_date")):
                continue
            ev_points.append(
                {
                    "date": pd.Timestamp(r["effective_trading_date"]).strftime("%Y-%m-%d"),
                    "category": r.get("category"),
                    "title": str(r.get("description", ""))[:120],
                    "car5": None if pd.isna(r.get("car_t5")) else round(float(r["car_t5"]), 4),
                }
            )

    sent_dates, sent_vals = [], []
    if sentiment is not None and not sentiment.empty and "avg_sentiment" in sentiment.columns:
        s = sentiment.dropna(subset=["avg_sentiment"]).tail(400)
        sent_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in s["date"]]
        sent_vals = [round(float(v), 4) for v in s["avg_sentiment"]]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PSX Event & Sentiment Intelligence</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #172029;
    --text: #e7eef6;
    --muted: #8aa0b5;
    --accent: #2 thrash;
  }}
  body {{ margin:0; font-family: "Segoe UI", Tahoma, sans-serif; background: linear-gradient(160deg,#0f1419,#182230 50%,#101820); color: var(--text); }}
  header {{ padding: 28px 32px 8px; }}
  h1 {{ margin:0; font-size: 28px; letter-spacing: .02em; }}
  p.sub {{ color: var(--muted); margin: 8px 0 0; max-width: 820px; }}
  .grid {{ display:grid; grid-template-columns: 1fr; gap: 16px; padding: 16px 24px 40px; }}
  .card {{ background: rgba(23,32,41,.92); border: 1px solid #243141; border-radius: 14px; padding: 8px; }}
  .note {{ color: var(--muted); font-size: 13px; padding: 0 32px 24px; }}
</style>
</head>
<body>
<header>
  <h1>PSX Event & Sentiment Intelligence</h1>
  <p class="sub">KSE-100 proxy, curated national events, and reactionary sentiment where text is available. Research dashboard — not trading advice.</p>
</header>
<div class="grid">
  <div class="card"><div id="price"></div></div>
  <div class="card"><div id="sentiment"></div></div>
  <div class="card"><div id="events"></div></div>
</div>
<p class="note">Event markers use investor information dates mapped to the next available trading session. Historical sentiment is shown only when articles were scored; gaps are intentional.</p>
<script>
const dates = {mkt_dates};
const closes = {closes};
const vols = {vols};
const events = {ev_points};
const sentDates = {sent_dates};
const sentVals = {sent_vals};

Plotly.newPlot('price', [{{
  x: dates, y: closes, type: 'scatter', mode: 'lines', name: 'KSE-100 proxy',
  line: {{color: '#6ec1ff', width: 2}}
}}, {{
  x: events.map(e => e.date),
  y: events.map(e => closes[dates.indexOf(e.date)] ?? null),
  text: events.map(e => e.category + ': ' + e.title),
  mode: 'markers', name: 'Events',
  marker: {{size: 9, color: '#ffb454'}}
}}], {{
  title: 'PSX level with event markers',
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {{color: '#e7eef6'}},
  xaxis: {{gridcolor: '#243141'}}, yaxis: {{gridcolor: '#243141', title: 'Index'}}
}}, {{responsive: true}});

Plotly.newPlot('sentiment', [{{
  x: sentDates, y: sentVals, type: 'scatter', mode: 'lines+markers', name: 'Avg sentiment',
  line: {{color: '#7ddea2'}}
}}], {{
  title: 'Daily sentiment (observed days only)',
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {{color: '#e7eef6'}},
  xaxis: {{gridcolor: '#243141'}}, yaxis: {{range: [-1,1], gridcolor: '#243141'}}
}}, {{responsive: true}});

Plotly.newPlot('events', [{{
  x: events.map(e => e.date),
  y: events.map(e => e.car5),
  text: events.map(e => e.title),
  type: 'bar',
  marker: {{color: events.map(e => (e.car5 ?? 0) >= 0 ? '#3ecf8e' : '#f07178')}}
}}], {{
  title: 'Event CAR (T to T+5)',
  paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  font: {{color: '#e7eef6'}},
  xaxis: {{gridcolor: '#243141'}}, yaxis: {{tickformat: '.1%', gridcolor: '#243141'}}
}}, {{responsive: true}});
</script>
</body>
</html>
"""
    # Fix accidental typo in CSS
    html = html.replace("--accent: #2 thrash;", "--accent: #6ec1ff;")
    out.write_text(html, encoding="utf-8")
    return out
