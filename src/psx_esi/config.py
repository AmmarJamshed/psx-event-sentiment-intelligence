from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, rel in list(cfg.get("paths", {}).items()):
        p = Path(rel)
        if not p.is_absolute():
            cfg["paths"][key] = str(ROOT / p)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key, value in cfg.get("paths", {}).items():
        if key.endswith("_dir") or key in {"figures_dir", "reports_dir", "models_dir", "output_dir"}:
            Path(value).mkdir(parents=True, exist_ok=True)
