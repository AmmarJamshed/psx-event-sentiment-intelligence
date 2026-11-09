"""Build a plain-language category impact dashboard (HTML)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
mapped = pd.read_csv(ROOT / "output" / "event_study" / "event_market_map.csv")
for c in ("car_t5", "ret_t", "car_t0", "car_t1", "car_t3", "car_t10", "car_t20", "impact_score"):
    if c in mapped.columns:
        mapped[c] = pd.to_numeric(mapped[c], errors="coerce")

mapped["category"] = mapped["category"].astype(str).str.lower().str.strip()
mapped["short"] = mapped["description"].fillna("").astype(str).str.slice(0, 110)

cat = (
    mapped.groupby("category")
    .agg(
        n=("event_id", "count"),
        mean_abs_car5=("car_t5", lambda s: s.abs().mean()),
        mean_car5=("car_t5", "mean"),
        mean_abs_day0=("ret_t", lambda s: s.abs().mean()),
        mean_abs_car20=("car_t20", lambda s: s.abs().mean()),
    )
    .reset_index()
    .sort_values("mean_abs_car5", ascending=False)
)
winner = cat.iloc[0]
labels = {
    "economic": "Money / economy news",
    "geopolitical": "War / security / borders",
    "political": "Politics / elections",
    "social": "Disasters / society",
    "regulatory": "Rules / FATF / regulators",
}
cat["label"] = cat["category"].map(labels).fillna(cat["category"])
cat["abs_pct"] = (cat["mean_abs_car5"] * 100).round(2)
cat["signed_pct"] = (cat["mean_car5"] * 100).round(2)

worst = mapped.nsmallest(8, "car_t5")[
    ["event_date", "category", "car_t5", "ret_t", "short"]
].copy()
worst["event_date"] = pd.to_datetime(worst["event_date"]).dt.strftime("%Y-%m-%d")
worst["week_pct"] = (worst["car_t5"] * 100).round(2)
worst["day_pct"] = (worst["ret_t"] * 100).round(2)
worst["label"] = worst["category"].map(labels).fillna(worst["category"])

horizons = []
for key, name in [
    ("car_t0", "Same day"),
    ("car_t1", "1 day later"),
    ("car_t3", "3 days later"),
    ("car_t5", "1 week later"),
    ("car_t10", "2 weeks later"),
    ("car_t20", "1 month later"),
]:
    if key in mapped.columns:
        horizons.append({"name": name, "abs_pct": round(float(mapped[key].abs().mean() * 100), 2)})

payload = {
    "n_events": int(len(mapped)),
    "winner": {
        "category": winner["category"],
        "label": labels.get(winner["category"], winner["category"]),
        "abs_pct": round(float(winner["mean_abs_car5"] * 100), 2),
        "n": int(winner["n"]),
    },
    "categories": cat.to_dict(orient="records"),
    "worst": worst.to_dict(orient="records"),
    "horizons": horizons,
}

out_json = ROOT / "output" / "dashboard" / "category_impact.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# Simple human labels for bars
bars = ""
max_abs = max(c["abs_pct"] for c in payload["categories"]) or 1
for i, c in enumerate(payload["categories"]):
    width = max(8, int(c["abs_pct"] / max_abs * 100))
    medal = "🥇 " if i == 0 else ("🥈 " if i == 1 else ("🥉 " if i == 2 else ""))
    bars += f"""
    <div class="bar-row">
      <div class="bar-label">{medal}{c['label']}<span>{int(c['n'])} events</span></div>
      <div class="bar-track"><div class="bar-fill {'winner' if i==0 else ''}" style="width:{width}%"></div></div>
      <div class="bar-val">{c['abs_pct']:.2f}%</div>
    </div>"""

worst_rows = ""
for i, r in enumerate(payload["worst"], 1):
    worst_rows += f"""
    <tr>
      <td>{i}</td>
      <td>{r['event_date']}</td>
      <td><span class="chip {r['category']}">{r['label']}</span></td>
      <td class="neg">{r['week_pct']:.2f}%</td>
      <td class="neg">{r['day_pct']:.2f}%</td>
      <td>{r['short']}…</td>
    </tr>"""

hz_bars = ""
max_h = max(h["abs_pct"] for h in horizons) or 1
for h in horizons:
    w = max(6, int(h["abs_pct"] / max_h * 100))
    hz_bars += f"""
    <div class="bar-row compact">
      <div class="bar-label">{h['name']}</div>
      <div class="bar-track"><div class="bar-fill soft" style="width:{w}%"></div></div>
      <div class="bar-val">{h['abs_pct']:.2f}%</div>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PSX Event Impact — Simple Dashboard</title>
<style>
  :root {{
    --bg: #f4f1ea;
    --ink: #1c1917;
    --muted: #57534e;
    --card: #fffdf8;
    --line: #e7e0d4;
    --red: #b42318;
    --green: #067647;
    --blue: #175cd3;
    --amber: #b54708;
    --purple: #6941c6;
    --teal: #0e7490;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    background:
      radial-gradient(1200px 500px at 10% -10%, #ffe8cc 0%, transparent 55%),
      radial-gradient(900px 400px at 100% 0%, #dbeafe 0%, transparent 50%),
      var(--bg);
    color: var(--ink);
    line-height: 1.45;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 64px; }}
  .hero {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 28px 28px 24px;
    margin-bottom: 18px;
  }}
  .hero h1 {{
    margin: 0 0 8px;
    font-size: clamp(1.8rem, 4vw, 2.6rem);
    letter-spacing: -0.03em;
    line-height: 1.1;
  }}
  .hero .answer {{
    font-size: clamp(1.35rem, 3vw, 2rem);
    font-weight: 800;
    color: var(--amber);
    margin: 10px 0 6px;
  }}
  .hero p {{ margin: 0; color: var(--muted); max-width: 46rem; }}
  .grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; }}
  @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
  }}
  .card h2 {{ margin: 0 0 6px; font-size: 1.15rem; }}
  .card .hint {{ color: var(--muted); font-size: 0.95rem; margin: 0 0 16px; }}
  .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px; }}
  @media (max-width: 700px) {{ .stats {{ grid-template-columns: 1fr; }} }}
  .stat {{
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
    background: #fff;
  }}
  .stat .k {{ color: var(--muted); font-size: 0.85rem; }}
  .stat .v {{ font-size: 1.45rem; font-weight: 800; margin-top: 4px; }}
  .bar-row {{ display: grid; grid-template-columns: 210px 1fr 64px; gap: 10px; align-items: center; margin: 10px 0; }}
  .bar-row.compact {{ grid-template-columns: 120px 1fr 56px; }}
  .bar-label {{ font-weight: 650; font-size: 0.95rem; }}
  .bar-label span {{ display:block; font-weight: 500; color: var(--muted); font-size: 0.8rem; }}
  .bar-track {{ height: 18px; background: #eee7db; border-radius: 999px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: var(--blue); border-radius: 999px; }}
  .bar-fill.winner {{ background: var(--amber); }}
  .bar-fill.soft {{ background: var(--teal); }}
  .bar-val {{ text-align: right; font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 650; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em; }}
  .neg {{ color: var(--red); font-weight: 750; white-space: nowrap; }}
  .chip {{
    display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.78rem; font-weight: 650;
    background: #f5f5f4; color: #44403c;
  }}
  .chip.economic {{ background: #ffedd5; color: #9a3412; }}
  .chip.geopolitical {{ background: #fee2e2; color: #991b1b; }}
  .chip.political {{ background: #e0e7ff; color: #3730a3; }}
  .chip.social {{ background: #dcfce7; color: #166534; }}
  .chip.regulatory {{ background: #f3e8ff; color: #6b21a8; }}
  .story {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 8px;
  }}
  @media (max-width: 860px) {{ .story {{ grid-template-columns: 1fr; }} }}
  .story .box {{
    border: 1px solid var(--line); border-radius: 14px; padding: 14px; background: #fff;
  }}
  .story .box strong {{ display:block; margin-bottom: 6px; font-size: 1.02rem; }}
  .footnote {{ color: var(--muted); font-size: 0.85rem; margin-top: 8px; }}
  .ing {{ display:grid; grid-template-columns: 170px 1fr 58px; gap: 10px; align-items:center; margin: 8px 0; }}
  .ing .name {{ font-weight: 650; font-size: 0.92rem; }}
  .ing .track {{ height: 16px; background:#eee7db; border-radius:999px; overflow:hidden; }}
  .ing .fill {{ height:100%; border-radius:999px; background: var(--blue); }}
  .ing .fill.big {{ background: var(--amber); }}
  .ing .pct {{ text-align:right; font-weight:700; }}
  .school {{ display:grid; grid-template-columns: repeat(4,1fr); gap:8px; margin-top:10px; }}
  @media (max-width: 860px) {{ .school {{ grid-template-columns: 1fr 1fr; }} }}
  .grade {{ border:1px solid var(--line); border-radius:12px; padding:10px; background:#fff; text-align:center; }}
  .grade .g {{ font-size:1.35rem; font-weight:800; }}
  .grade .m {{ font-size:0.82rem; color:var(--muted); }}
  .paths {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px; }}
  @media (max-width: 860px) {{ .paths {{ grid-template-columns:1fr; }} }}
  .upbox, .downbox {{ border:1px solid var(--line); border-radius:14px; padding:14px; background:#fff; }}
  .upbox {{ border-left:5px solid var(--green); }}
  .downbox {{ border-left:5px solid var(--red); }}
  .upbox h3 {{ margin:0 0 8px; color:var(--green); font-size:1.05rem; }}
  .downbox h3 {{ margin:0 0 8px; color:var(--red); font-size:1.05rem; }}
  .upbox ul, .downbox ul {{ margin:0; padding-left:1.1rem; }}
  .upbox li, .downbox li {{ margin:6px 0; }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.78rem; font-weight:700; }}
  .tag.up {{ background:#dcfce7; color:#166534; }}
  .tag.down {{ background:#fee2e2; color:#991b1b; }}
  .tag.mix {{ background:#ffedd5; color:#9a3412; }}
  .bubble {{ background:#fff; border:1px dashed var(--line); border-radius:14px; padding:12px 14px; margin-top:10px; }}
</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>What shakes Pakistan’s stock market?</h1>
      <div class="answer">Answer: {payload['winner']['label']} moves it most.</div>
      <p>
        We studied <strong>{payload['n_events']} real national events</strong> and measured how much the KSE-100
        moved in the week after each one. Bigger bar = bigger typical market shake.
      </p>
      <div class="stats">
        <div class="stat"><div class="k">Winner category</div><div class="v">{payload['winner']['label']}</div></div>
        <div class="stat"><div class="k">Average week move</div><div class="v">{payload['winner']['abs_pct']:.2f}%</div></div>
        <div class="stat"><div class="k">Events in sample</div><div class="v">{payload['n_events']}</div></div>
      </div>
    </section>

    <div class="grid">
      <section class="card">
        <h2>1. Which event type shakes stocks most?</h2>
        <p class="hint">Average size of the 1-week market move (up or down). This is the main score.</p>
        {bars}
        <p class="footnote">Source: PSX event study · metric = mean |CAR T→T+5| · association, not proven causation.</p>
      </section>
      <section class="card">
        <h2>2. How long do shocks last?</h2>
        <p class="hint">Average move size grows as more days pass — news is not finished on day one.</p>
        {hz_bars}
        <p class="footnote">Mean absolute cumulative return after the event trading day.</p>
      </section>
    </div>

    <section class="card">
      <h2>3. Biggest losing weeks (examples)</h2>
      <p class="hint">These events were followed by the largest one-week drops in the study.</p>
      <div style="overflow-x:auto">
      <table>
        <thead>
          <tr><th>#</th><th>Date</th><th>Type</th><th>Week</th><th>Day 0</th><th>What happened</th></tr>
        </thead>
        <tbody>
          {worst_rows}
        </tbody>
      </table>
      </div>
    </section>

    <section class="card">
      <h2>4. The relationship in plain words</h2>
      <div class="story">
        <div class="box">
          <strong>Money stress hurts most</strong>
          FX crises, IMF stress, rate shocks, and inflation/energy pain create the largest typical weekly swings.
        </div>
        <div class="box">
          <strong>Security scares hit fast</strong>
          India tensions and terror attacks often create sharp risk-off weeks (sometimes the single worst week).
        </div>
        <div class="box">
          <strong>Politics matters, but less on average</strong>
          Elections and leadership fights move stocks, yet the average weekly shake is smaller than economic stress.
        </div>
      </div>
      <p class="footnote">
        Simple rule: <em>Event type + how bad it is → stock move over the next week</em>, after checking the next trading day
        (weekends/holidays). Correlation is not the same as causation.
      </p>
    </section>

    <section class="card">
      <h2>5. Computer recipe (regression / ML)</h2>
      <p class="hint">Kid version: the computer cooks a “guess” of the stock move from ingredients. Bigger bar = more useful ingredient.</p>
      <div class="stats" style="margin-bottom:12px">
        <div class="stat"><div class="k">Best same-day robot</div><div class="v">Random Forest</div></div>
        <div class="stat"><div class="k">Up/down accuracy</div><div class="v">~60%</div></div>
        <div class="stat"><div class="k">Main ingredients</div><div class="v">Trend + volatility</div></div>
      </div>
      <div class="ing"><div class="name">Recent market trend</div><div class="track"><div class="fill big" style="width:100%"></div></div><div class="pct">29%</div></div>
      <div class="ing"><div class="name">Yesterday’s return</div><div class="track"><div class="fill big" style="width:64%"></div></div><div class="pct">19%</div></div>
      <div class="ing"><div class="name">Price jumpiness</div><div class="track"><div class="fill" style="width:51%"></div></div><div class="pct">15%</div></div>
      <div class="ing"><div class="name">Recent volatility</div><div class="track"><div class="fill" style="width:44%"></div></div><div class="pct">13%</div></div>
      <div class="ing"><div class="name">Emerging markets</div><div class="track"><div class="fill" style="width:18%"></div></div><div class="pct">5%</div></div>
      <div class="ing"><div class="name">Oil / US stocks / PKR</div><div class="track"><div class="fill" style="width:16%"></div></div><div class="pct">~5%</div></div>
      <div class="school">
        <div class="grade"><div class="g">A-</div><div class="m">Random Forest</div></div>
        <div class="grade"><div class="g">B</div><div class="m">Lasso</div></div>
        <div class="grade"><div class="g">B-</div><div class="m">Elastic Net</div></div>
        <div class="grade"><div class="g">C</div><div class="m">OLS / Ridge</div></div>
      </div>
      <div class="bubble">
        <strong>Remember:</strong> day-to-day, the model mostly watches what the market was already doing + world prices.
        Big economy/security events still cause the largest <em>weekly</em> shakes (sections 1–3).
      </div>
    </section>

    <section class="card">
      <h2>6. Recent news — up or down lean?</h2>
      <p class="hint">From the latest Dawn Business headlines scored in this project (soft weather report, not a trade signal).</p>
      <div style="overflow-x:auto">
      <table>
        <thead><tr><th>Headline</th><th>Lean</th><th>Why it matters</th></tr></thead>
        <tbody>
          <tr><td>Pakistan–Saudi investors launch investment fund</td><td><span class="tag up">UP</span></td><td>Gulf capital historically supports FX confidence.</td></tr>
          <tr><td>IT earnings drive 22.6% growth in services exports</td><td><span class="tag up">UP</span></td><td>Export earnings help the external account.</td></tr>
          <tr><td>NEPRA raises special industrial tariff</td><td><span class="tag down">DOWN</span></td><td>Higher power costs squeeze industry margins.</td></tr>
          <tr><td>LPG reserved for SOEs / power-sector audit demand</td><td><span class="tag down">DOWN</span></td><td>Energy & circular-debt stress = classic economy risk.</td></tr>
          <tr><td>Riba deadline uncertainty / SECP IPO probe</td><td><span class="tag mix">MIX</span></td><td>Regulatory fog can pause financials.</td></tr>
          <tr><td>Ports busy on ME disruption, need road reforms</td><td><span class="tag mix">MIX</span></td><td>Near-term traffic upside vs reform lag.</td></tr>
        </tbody>
      </table>
      </div>
      <p class="footnote">Snapshot sentiment was near neutral (~0.03). Near term looks like a tug-of-war.</p>
    </section>

    <section class="card">
      <h2>7. What could change the path next?</h2>
      <div class="paths">
        <div class="upbox">
          <h3>Could push PSX UP</h3>
          <ul>
            <li><strong>Smooth IMF reviews</strong> — unlocks money + calms the rupee</li>
            <li><strong>Stable / stronger PKR</strong></li>
            <li><strong>Real Gulf FDI / deposits</strong> (Saudi/UAE style inflows)</li>
            <li><strong>Soft inflation + careful rate cuts</strong></li>
            <li><strong>Export / remittance strength</strong></li>
            <li><strong>Quiet geopolitics</strong></li>
          </ul>
        </div>
        <div class="downbox">
          <h3>Could push PSX DOWN</h3>
          <ul>
            <li><strong>PKR crash / reserve stress</strong></li>
            <li><strong>IMF programme slip</strong></li>
            <li><strong>Energy tariff / circular-debt flare</strong></li>
            <li><strong>Political rupture</strong></li>
            <li><strong>India / security escalation</strong></li>
            <li><strong>Oil spike + global risk-off</strong></li>
          </ul>
        </div>
      </div>
      <p class="footnote">Research prototype only. Association ≠ causation. Not investment advice.</p>
    </section>
  </div>
</body>
</html>
"""

out_html = ROOT / "output" / "dashboard" / "simple_impact.html"
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html}")
print(f"Wrote {out_json}")
print("WINNER", payload["winner"])
