from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = {
    "event_id",
    "event_date",
    "announcement_date",
    "category",
    "title",
    "description",
    "source",
    "verification_status",
    "severity_prior",
    "expected_direction_prior",
}


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "events" in raw:
        records = raw["events"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unrecognized events schema in {path}")

    df = pd.DataFrame(records)
    missing = REQUIRED_FIELDS - set(df.columns)
    if missing:
        raise ValueError(f"Events file missing required fields: {sorted(missing)}")

    for col in ("event_date", "announcement_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["severity_prior"] = pd.to_numeric(df["severity_prior"], errors="coerce")
    df = df.sort_values(["announcement_date", "event_id"]).reset_index(drop=True)
    return df


def events_summary(df: pd.DataFrame) -> dict:
    return {
        "n_events": int(len(df)),
        "date_min": str(df["event_date"].min().date()) if len(df) else None,
        "date_max": str(df["event_date"].max().date()) if len(df) else None,
        "by_category": df["category"].value_counts().to_dict(),
        "by_verification": df["verification_status"].value_counts().to_dict(),
    }
