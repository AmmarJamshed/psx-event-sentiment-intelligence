# PSX Historical Events Database

Verified (not fabricated) catalogue of Pakistan events used for equity-market / investor-confidence research around the Pakistan Stock Exchange (PSX).

## File

- `historical_events.json` — JSON **array** of event objects (2000–2025)

## Schema

Each object has **exactly** these fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | string | Stable ID, e.g. `EVT-2008-001` |
| `event_date` | string | `YYYY-MM-DD` — occurrence or first trading-relevant date |
| `announcement_date` | string | `YYYY-MM-DD` — when investors likely learned (often same or next session) |
| `event_time` | string | One of: `pre-market`, `during-session`, `after-close`, `unknown` |
| `category` | string | One of: `political`, `economic`, `geopolitical`, `regulatory`, `social`, `other` |
| `subcategory` | string | Short topic label |
| `title` | string | Concise headline |
| `description` | string | 2–4 factual sentences |
| `source` | string | Outlet or institution name |
| `source_url` | string | Real URL if known; otherwise `""` |
| `verification_status` | string | `high` (widely documented) or `medium` (date/process approximate) |
| `severity_prior` | integer | Expert prior of market relevance, 1–10 (not measured impact) |
| `expected_direction_prior` | string | Prior expectation only: `positive`, `negative`, `mixed`, `uncertain` |
| `tags` | string[] | Free-form topic tags |
| `notes` | string | Date uncertainty, caveats, or event-study guidance |

## Integrity policy

1. **No invented events** — only historically documented developments.
2. **No invented market outcomes** — do not store or infer returns, volumes, or sentiment scores in this file.
3. **No invented URLs** — if a precise primary link is not known, `source_url` must be `""`.
4. **Priors ≠ outcomes** — `severity_prior` and `expected_direction_prior` are research priors, not claimed realised PSX moves.
5. **Approximate dates** — use `verification_status: "medium"` and explain in `notes` (especially multi-week FX crises, IMF sequencing, or composite themes).
6. **Event studies** — for tight windows, prefer `high` rows; for `medium` rows, pin the date to a primary source before estimation.

## Coverage intent

Political (elections, no-confidence, leadership/judicial shocks), economic (IMF, SBP, PKR, budgets, circular debt), geopolitical (India, terror, Afghanistan, major powers), regulatory (FATF, SECP/PSX, privatisation, tax), and social (disasters, strikes/fuel/energy stress).

## Maintenance

When adding events: keep the exact field set, never fabricate URLs or market stats, and re-check year/category balance for research sampling.
