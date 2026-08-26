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
import hashlib
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
    """Quarantined feature names and the availability classes in force.

    Read from ``artifacts/features/quarantined_features.json`` - the manifest
    the release gate tests the bundle against - and not from
    ``configs/leakage_quarantine.yaml``. The YAML predates the Feature
    Availability Firewall and lists 4 columns where the firewall quarantines
    13, so the registry has until now been publishing a provenance block that
    understated its own quarantine by nine columns, four of which
    (``F3898``, ``F3913``, ``F3914``, ``F3916``) are precisely the ones the
    superseded CatBoost bundle had selected. A provenance record that
    under-reports the policy it was built under is worse than no record.
    """
    names: list[str] = []
    try:
        man = load_json(settings.FEATURES_DIR / "quarantined_features.json")
        names = [str(e["feature"]) for e in man.get("quarantine", [])
                 if e.get("feature")]
    except (FileNotFoundError, ValueError):
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
    """The feature set the shipped bundle actually carries.

    The bundle, not a selector artifact. The previous version walked
    ``final_selected_features.json`` and fell through to ``len(d)`` on a dict
    that has six top-level keys (``generated_utc``, ``method``, ``selector``,
    ``n_repeats_used``, ``pools``, ``__provenance__``) - so it published
    ``n_features: 6`` for a 120-feature model. That file is a record of the
    per-view selection *pools*, not of what shipped; only the bundle knows
    which pool won and which columns survived into it.
    """
    from muleguard.models.scoring import load_bundle

    p = settings.MODELS_DIR / "final_bundle.joblib"
    try:
        kept = list(load_bundle()["feature_list_kept"])
    except Exception as exc:  # noqa: BLE001
        return {"source": None,
                "note": f"model bundle unreadable, so the shipped feature set "
                        f"could not be established: {exc}"}
    return {
        "source": "artifacts/models/final_bundle.joblib:feature_list_kept",
        "bundle_sha256": sha256_file(p) if p.exists() else None,
        # Hashed the way muleguard.usp.baseline._feature_hash hashes it, so
        # the registry's number and the regression baseline's number are
        # comparable by eye rather than only by re-deriving one of them.
        "feature_list_sha256": hashlib.sha256(
            chr(10).join(kept).encode("utf-8")).hexdigest(),
        "n_features": len(kept),
    }


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
