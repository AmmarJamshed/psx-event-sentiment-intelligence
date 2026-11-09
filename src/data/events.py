from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.utils import ROOT, ensure_dirs, load_config


REQUIRED_COLS = [
    "event_id",
    "event_date",
    "event_time",
    "info_date",
    "category",
    "subcategory",
    "description",
    "source",
    "source_url",
    "severity_score",
    "verified",
]


def _normalize_event_records(rows: list[dict]) -> pd.DataFrame:
    """Accept either seed schema or historical_events.json research schema."""
    mapped = []
    for r in rows:
        severity = r.get("severity_score", r.get("severity_prior", 5))
        try:
            severity_f = float(severity)
        except (TypeError, ValueError):
            severity_f = 5.0
        # historical_events uses 1-10 priors; seed uses 0-1
        if severity_f > 1.0:
            severity_f = severity_f / 10.0
        verified = r.get("verified")
        if verified is None:
            verified = str(r.get("verification_status", "")).lower() == "high"
        mapped.append(
            {
                "event_id": r.get("event_id"),
                "event_date": r.get("event_date"),
                "event_time": r.get("event_time", "unknown"),
                "info_date": r.get("info_date") or r.get("announcement_date") or r.get("event_date"),
                "category": str(r.get("category", "other")).lower(),
                "subcategory": r.get("subcategory", ""),
                "title": r.get("title", ""),
                "description": r.get("description", ""),
                "source": r.get("source", ""),
                "source_url": r.get("source_url", ""),
                "severity_score": severity_f,
                "expected_direction_prior": r.get("expected_direction_prior", "uncertain"),
                "verification_status": r.get("verification_status", "high" if verified else "medium"),
                "verified": bool(verified),
                "notes": r.get("notes", ""),
            }
        )
    df = pd.DataFrame(mapped)
    for col in ("event_date", "info_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.sort_values("event_date").reset_index(drop=True)


def seed_events_if_missing(force: bool = False) -> Path:
    paths = ensure_dirs()
    dest = paths["events"] / "events.csv"
    # Prefer the comprehensive researched database when present.
    primary = ROOT / "data" / "events" / "historical_events.json"
    seed = ROOT / "data" / "events" / "events_seed.json"
    source = primary if primary.exists() else seed
    if dest.exists() and not force:
        # Rebuild if historical DB is newer / larger intent: always refresh from primary JSON when present
        if primary.exists() and dest.stat().st_mtime < primary.stat().st_mtime:
            force = True
        else:
            return dest
    with open(source, encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw["events"] if isinstance(raw, dict) and "events" in raw else raw
    df = _normalize_event_records(rows)
    df.to_csv(dest, index=False)
    shutil.copy(source, paths["output"] / "events" / source.name)
    return dest


def load_events(path: Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    csv_path = path or seed_events_if_missing()
    df = pd.read_csv(csv_path, parse_dates=["event_date", "info_date"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Event database missing columns: {missing}")
    df["category"] = df["category"].str.lower().str.strip()
    df["verified"] = df["verified"].astype(bool)
    return df.sort_values("event_date").reset_index(drop=True)


def export_events(df: pd.DataFrame) -> Path:
    paths = ensure_dirs()
    out = paths["output"] / "events" / "events.csv"
    df.to_csv(out, index=False)
    return out
