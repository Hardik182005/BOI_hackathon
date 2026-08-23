"""Fit and freeze the Cohort Radar similarity transform.

Run::

    .venv/Scripts/python.exe -m muleguard.cli.build_cohort_radar

Writes:

* ``artifacts/models/cohort_radar_transform.joblib`` - the frozen transform
* ``artifacts/models/cohort_radar_manifest.json`` - what it was built from

Nothing here trains, tunes or touches the classifier. The command reads the
development partition, derives scaling statistics and an empirical null
similarity distribution, and writes both to disk so every later query is
answered against the same yardstick.

The manifest is the point of the exercise. A similarity score is only auditable
if a reader can check what produced it: which features, which weights and where
they came from, which quarantine was in force, what the null distribution looked
like, and which commit built it. All of that is recorded, hashed where a hash is
meaningful.

Rebuilding is safe and idempotent - the fit is seeded - but it is still a
release-visible act, because the frozen percentile bands are what turn a raw
similarity into the word VERY_HIGH on an analyst screen.
"""
from __future__ import annotations

import argparse

import numpy as np

from muleguard import settings
from muleguard.logging import configure, get_logger
from muleguard.usp import cohort_radar as radar
from muleguard.utils import git_info, save_json, sha256_file

log = get_logger("cli.build_cohort_radar")


def _label_conditioned_null(transform, ref_frame, rows) -> dict:
    """The same null, restricted to pairs of negative accounts.

    Section 12 suggests sampling legitimate account pairs. The frozen bands are
    built label-free instead, so that the transform is provably independent of
    F3924 and the leakage tests in section 19 pass by construction rather than
    by argument. This function computes the label-conditioned version purely as
    a diagnostic, so the claim that the two nulls agree is a measurement anyone
    can check rather than an assertion in a document.
    """
    from muleguard.features.frame import raw_with_meta

    y = raw_with_meta()[rows.tolist()][settings.TARGET_COLUMN].to_numpy()
    negatives = np.flatnonzero(np.nan_to_num(y, nan=0.0) == 0)
    if len(negatives) < 100:
        return {"computed": False, "reason": "too few negative rows to sample"}

    num, cat = transform.encode(ref_frame)
    rng = np.random.default_rng(transform.seed)
    n_pairs = min(transform.n_null_pairs, 200000)
    left = negatives[rng.integers(0, len(negatives), size=n_pairs)]
    right = negatives[rng.integers(0, len(negatives), size=n_pairs)]
    keep = left != right
    left, right = left[keep], right[keep]

    sims = np.empty(len(left))
    for start in range(0, len(left), 20000):
        sl = slice(start, start + 20000)
        sims[sl] = transform.similarity_aligned(
            num[left[sl]], cat[left[sl]], num[right[sl]], cat[right[sl]])

    bands = {name: float(np.percentile(sims, pct))
             for name, pct in transform.band_percentiles.items()}
    deltas = {name: abs(bands[name] - transform.bands[name]) for name in bands}
    return {
        "computed": True,
        "purpose": ("diagnostic only - the frozen bands are the label-free ones; "
                    "this shows what conditioning on the label would have given"),
        "n_negative_rows": int(len(negatives)),
        "n_pairs_used": int(len(left)),
        "mean": float(sims.mean()),
        "bands_if_label_conditioned": bands,
        "bands_actually_frozen": dict(transform.bands),
        "max_band_difference": float(max(deltas.values())) if deltas else 0.0,
    }


def build(*, with_diagnostic: bool = True) -> dict:
    from muleguard.features import firewall
    from muleguard.features.frame import raw_with_meta

    transform = radar.fit()
    sha = radar.save(transform)

    rows = radar.reference_row_index()
    ref_frame = raw_with_meta()[rows.tolist()].select(transform.features)
    diagnostic = (_label_conditioned_null(transform, ref_frame, rows)
                  if with_diagnostic else {"computed": False, "reason": "skipped"})

    cat_w = transform.categorical_weights
    cfg = firewall.config()
    manifest = {
        "generated_utc": transform.fitted_utc,
        "radar_version": transform.radar_version,
        "git": git_info(settings.REPO_ROOT),
        "transform_path": "artifacts/models/cohort_radar_transform.joblib",
        "transform_sha256": sha,
        "source_data": {
            "dataset_sha256": sha256_file(settings.REPO_ROOT / "DataSet.xlsx"),
            "reference_partition": "development split only; locked test excluded",
            "n_reference_rows": transform.n_reference_rows,
        },
        "fingerprint": {
            "n_features": len(transform.features),
            "n_numeric": len(transform.numeric_features),
            "n_categorical": len(transform.categorical_features),
            "features": transform.features,
            "feature_list_hash": transform.feature_hash(),
            "source": ("the selected feature list of the champion model, "
                       "re-admitted through the leakage firewall at build time"),
        },
        "quarantine": {
            "policy_version": cfg.policy_version,
            "quarantine_hash": transform.quarantine_hash,
            "target_column": settings.TARGET_COLUMN,
            "target_in_fingerprint": settings.TARGET_COLUMN in transform.features,
        },
        "scaling": {
            "statistics_hash": transform.scaling_hash(),
            "numeric_scale": "min(|x-y| / (4*IQR), 1), IQR from development rows",
            "scale_source_counts": {
                s: transform.scale_source.count(s)
                for s in sorted(set(transform.scale_source))},
            "clip_quantiles": [0.01, 0.99],
        },
        "weights": {
            "hash": transform.weights_hash(),
            **transform.weight_source,
            "sum": float(transform.numeric_weights.sum() + cat_w.sum()),
            "min": float(min(transform.numeric_weights.min(),
                             cat_w.min() if len(cat_w) else np.inf)),
            "max": float(max(transform.numeric_weights.max(),
                             cat_w.max() if len(cat_w) else -np.inf)),
        },
        "similarity_formula": (
            "S(x,y) = 1 - sum_j w_j * delta_j, clamped to [0,1]. "
            "Numeric delta_j = min(|x_j-y_j| / (4*IQR_j), 1). "
            "Categorical delta_j = 0 if the category matches else 1. "
            "Missing: both missing -> 0, exactly one missing -> 1. No imputation."),
        "null_distribution": transform.null_statistics,
        "percentile_thresholds": {
            "bands": transform.bands,
            "percentiles": transform.band_percentiles,
            "derived_from": ("empirical null over random development pairs; "
                             "never tuned on the locked test"),
        },
        "label_conditioned_null_diagnostic": diagnostic,
        "forbidden": {
            "edge_label": radar.EDGE_LABEL,
            "never_emitted": list(radar.FORBIDDEN_COHORT_LANGUAGE),
        },
    }
    save_json(manifest, radar.MANIFEST_PATH)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the Cohort Radar transform.")
    ap.add_argument("--skip-diagnostic", action="store_true",
                    help="skip the label-conditioned null comparison")
    args = ap.parse_args(argv)

    configure()
    manifest = build(with_diagnostic=not args.skip_diagnostic)
    t = manifest["fingerprint"]
    log.info("wrote %s", radar.TRANSFORM_PATH)
    log.info("wrote %s", radar.MANIFEST_PATH)
    log.info("  features            %d (%d numeric, %d categorical)",
             t["n_features"], t["n_numeric"], t["n_categorical"])
    log.info("  target in features  %s", manifest["quarantine"]["target_in_fingerprint"])
    log.info("  weight source       %s + %s", manifest["weights"].get("primary"),
             manifest["weights"].get("tail"))
    log.info("  weights sum to      %.12f", manifest["weights"]["sum"])
    log.info("  reference rows      %d", manifest["source_data"]["n_reference_rows"])
    null = manifest["null_distribution"]
    log.info("  null pairs          %d, mean similarity %.6f",
             null["n_pairs_used"], null["mean"])
    for name, value in manifest["percentile_thresholds"]["bands"].items():
        log.info("  %-22s >= %.6f (p%s)", name, value,
                 manifest["percentile_thresholds"]["percentiles"][name])
    diag = manifest["label_conditioned_null_diagnostic"]
    if diag.get("computed"):
        log.info("  label-conditioned null differs by at most %.6f on any band",
                 diag["max_band_difference"])
    if manifest["quarantine"]["target_in_fingerprint"]:
        log.error("target column is inside the fingerprint - refusing to ship")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
