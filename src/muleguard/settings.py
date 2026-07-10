"""Central paths, constants and configuration loading.

All pipeline code resolves paths and tunables through this module so that
every artifact lands in a known folder and every constant has one source
of truth (``configs/*.yaml``). Nothing here reads the dataset.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EVIDENCE_DIR = ARTIFACTS_DIR / "evidence"
REGISTRY_DIR = ARTIFACTS_DIR / "model_registry"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
FEATURES_DIR = ARTIFACTS_DIR / "features"
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"

DOCS_DIR = REPO_ROOT / "docs"

# Hard safety constants. configs/base.yaml must agree; ``load_config`` asserts it.
TARGET_COLUMN = "F3924"
GLOBAL_SEED = 42


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` (cached). ``name`` excludes extension."""
    cfg = _read_yaml(CONFIG_DIR / f"{name}.yaml")
    if name == "base":
        if cfg.get("target_column") != TARGET_COLUMN:
            raise ValueError(
                f"configs/base.yaml target_column={cfg.get('target_column')!r} "
                f"disagrees with settings.TARGET_COLUMN={TARGET_COLUMN!r}"
            )
        if cfg.get("seed") != GLOBAL_SEED:
            raise ValueError("configs/base.yaml seed disagrees with settings.GLOBAL_SEED")
    return cfg


def ensure_dirs() -> None:
    """Create every known artifact/data directory (idempotent)."""
    for d in (
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SPLITS_DIR,
        MODELS_DIR, METRICS_DIR, PLOTS_DIR, REPORTS_DIR, EVIDENCE_DIR,
        REGISTRY_DIR, PREDICTIONS_DIR, FEATURES_DIR, OPTUNA_DIR, DOCS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
