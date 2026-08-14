"""Add the provenance fields the spec requires to the model registry.

The registry recorded which model shipped and how it scored, but not enough to
answer "on what data, under which firewall, with which CV design?" - and a
registry that cannot answer those is a filename with a number beside it.

Every field written here is read from an existing frozen artifact. Nothing is
recomputed and no model is touched; where a value genuinely is not recoverable
from the artifacts, the field is written as null with a stated reason rather
than filled with a plausible guess.

    python -m muleguard.cli.enrich_registry
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import Any

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.utils import load_json, save_json, sha256_file

log = get_logger("cli.enrich_registry")

REGISTRY_PATH = settings.REGISTRY_DIR / "registry.json"


def _description_sha() -> tuple[str | None, str]:
    try:
        d = load_json(settings.FEATURES_DIR / "feature_dictionary.json")
        return d.get("source_sha256"), "artifacts/features/feature_dictionary.json"
    except (FileNotFoundError, ValueError):
        return None, "feature dictionary artifact absent"


def _quarantine() -> tuple[list[str], list[str]]:
    """Quarantined feature names and the availability classes in force."""
    import yaml

    names: list[str] = []
    try:
        with open(settings.CONFIG_DIR / "leakage_quarantine.yaml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        names = [str(e["feature"]) for e in cfg.get("quarantine", [])
                 if e.get("feature")]
    except FileNotFoundError:
        pass

    classes: list[str] = []
    try:
        d = load_json(settings.FEATURES_DIR / "feature_dictionary.json")
        classes = sorted(d.get("availability_counts", {}))
    except (FileNotFoundError, ValueError):
        pass
    return names, classes


def _cv_scheme() -> dict[str, Any]:
    splits = settings.load_config("data").get("splits", {})
    return {
        "outer": f"{splits.get('cv_n_repeats')}x{splits.get('cv_n_splits')} "
                 "repeated stratified k-fold",
        "n_splits": splits.get("cv_n_splits"),
        "n_repeats": splits.get("cv_n_repeats"),
        "locked_test_fraction": splits.get("locked_test_fraction"),
        "min_test_positives": splits.get("min_test_positives"),
        "grouped_by": "duplicate-row group id (no group spans a fold boundary)",
        "seed": settings.GLOBAL_SEED,
    }


def _holdout_metrics() -> dict[str, Any]:
    """Locked-test metrics, if they were already computed and stored.

    Never computed here: this module must not be a route by which the locked
    test gets read one more time.
    """
    for name in ("final_locked_test_metrics.json", "locked_test_metrics.json"):
        try:
            m = load_json(settings.METRICS_DIR / name)
        except (FileNotFoundError, ValueError):
            continue
        return {
            "source": f"artifacts/metrics/{name}",
            "pr_auc": m.get("pr_auc") or m.get("average_precision"),
            "brier": m.get("brier") or m.get("brier_score"),
            "ece": m.get("ece") or m.get("expected_calibration_error"),
        }
    return {"source": None,
            "note": "no stored locked-test metric file; not computed here "
                    "because this tool must never read the locked test"}


def _feature_set_version() -> dict[str, Any]:
    for name in ("final_selected_features.json", "selected_features.json"):
        p = settings.FEATURES_DIR / name
        if p.exists():
            try:
                d = load_json(p)
            except ValueError:
                continue
            n = (len(d) if isinstance(d, list)
                 else len(d.get("selected") or d.get("features") or d))
            return {"source": f"artifacts/features/{name}",
                    "sha256": sha256_file(p), "n_features": n}
    return {"source": None, "note": "no selected-feature artifact found"}


def enrich() -> dict[str, Any]:
    reg = load_json(REGISTRY_PATH)
    models = reg.get("models", [])
    if not models:
        raise SystemExit(f"{REGISTRY_PATH} lists no models")

    desc_sha, desc_src = _description_sha()
    quarantined, classes = _quarantine()

    provenance = {
        "description_sha256": desc_sha,
        "description_sha256_source": desc_src,
        "feature_set_version": _feature_set_version(),
        "availability_classes": classes,
        "quarantined_features": quarantined,
        "n_quarantined": len(quarantined),
        "cv_scheme": _cv_scheme(),
        "holdout_metrics": _holdout_metrics(),
        "enriched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": ("every field above is read from a frozen artifact; nothing "
                 "was recomputed and no model was touched"),
    }

    # The active model is the one a reader cares about; retired rows keep the
    # provenance that was true when they shipped, so they are left alone.
    active = [m for m in models if m.get("status") != "retired"]
    targets = active or models[-1:]
    for m in targets:
        m["provenance"] = provenance
    reg["registry_schema_version"] = "1.1"

    save_json(reg, REGISTRY_PATH)
    log.info("enriched %d active registry entr%s", len(targets),
             "y" if len(targets) == 1 else "ies")
    return provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    p = enrich()
    print(json.dumps({k: v for k, v in p.items() if k != "note"}, indent=2)[:1400])


if __name__ == "__main__":
    main()
