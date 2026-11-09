from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    paths = {k: ROOT / v for k, v in cfg["paths"].items()}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    for sub in (
        paths["output"] / "market",
        paths["output"] / "events",
        paths["output"] / "event_study",
        paths["output"] / "sentiment",
        paths["output"] / "models",
        paths["output"] / "reports",
        paths["output"] / "dashboard",
        paths["data_raw"] / "market",
        paths["data_raw"] / "news",
        paths["data_raw"] / "macro",
        paths["data_processed"],
        paths["events"],
    ):
        sub.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def next_trading_day(dates: pd.DatetimeIndex, event_date: pd.Timestamp) -> pd.Timestamp | None:
    """First trading session on or after event_date (handles weekends/holidays via available index)."""
    event_date = pd.Timestamp(event_date).normalize()
    later = dates[dates >= event_date]
    if len(later) == 0:
        return None
    return later[0]


def previous_trading_day(dates: pd.DatetimeIndex, event_date: pd.Timestamp) -> pd.Timestamp | None:
    event_date = pd.Timestamp(event_date).normalize()
    earlier = dates[dates < event_date]
    if len(earlier) == 0:
        return None
    return earlier[-1]
