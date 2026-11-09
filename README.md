# Pakistan Stock Market — Event & Sentiment Intelligence (PSX-ESI)

Quantitative research system linking **Pakistan national events**, **public/news sentiment**, and **PSX (KSE-100) returns**.

## Deliverables

| Deliverable | Location |
|---|---|
| A. Historical event database | `data/events/historical_events.json` (+ `events.csv`) |
| B. Event impact report | `output/event_study/` / `output/reports/` |
| C. Sentiment intelligence dashboard | `output/dashboard/` |
| D. Statistical research report | `output/reports/` |
| E. Predictive model artifacts | `output/models/` |
| F. Daily intelligence report | `output/reports/daily_intelligence.md` |

## Integrity rules

- Events are curated with sources; **market returns are measured from price data**, not invented.
- Historical sentiment is **not fabricated**. Only available news text is scored.
- Chronological train/validation/test splits — no random shuffle on time series.
- Correlation / event association ≠ causation.

## Quick start

```bat
cd D:\psx-event-sentiment-intelligence
.\.venv\Scripts\python.exe -m src.run_pipeline --all
```

Reuse cached market data:

```bat
.\.venv\Scripts\python.exe -m src.run_pipeline --all --no-refresh-market
```

## Pipeline modules (`src/`)

- `data/` — events, market (Yahoo `^KSE`), macro, news RSS
- `event_study.py` — multi-horizon CAR windows T…T+20
- `nlp/sentiment.py` — lexicon multi-emotion scoring (optional transformers)
- `features.py` — consolidated modelling panel
- `models.py` — OLS/Ridge/Lasso/ElasticNet/RF/GBM/XGBoost + VAR/ARIMAX
- `validation.py` — expanding/rolling windows, diagnostics, crisis vs stable
- `reports.py` — research report, Q&A, daily intel, dashboard

## Known data gaps

- Free KSE-100 history may be incomplete vs official PSX.
- Sector indices, advance/decline, and full market-cap breadth need licensed feeds.
- Dense historical Urdu/social sentiment requires archive access — gaps are reported, not filled with fake scores.

## Disclaimer

Research software only. Not investment advice.
