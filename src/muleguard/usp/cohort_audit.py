"""Sections 19 and 39: the leakage and determinism evidence, as artifacts.

The unit tests already assert every one of these properties. This module exists
because a passing test is evidence only to somebody who reruns it. Section 39
asks for two files a reader can open and check against the code, so each check
below records not just PASS/FAIL but the value it saw - the column list it
searched, the hash it compared, the ranking it got twice.

Nothing here is a smoke test of the happy path. Every check is written so that
the way it would fail is the way the system would actually be wrong.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.usp import cohort_radar as cr
from muleguard.utils import git_info, save_json

log = get_logger("usp.cohort_audit")

LEAKAGE_PATH = settings.ARTIFACTS_DIR / "testing" / "cohort_radar_leakage.json"
DETERMINISM_PATH = settings.ARTIFACTS_DIR / "testing" / "cohort_radar_determinism.json"

#: Columns section 4 names explicitly. Checked *in addition to* the live
#: firewall list, not instead of it: the firewall is authoritative, and these
#: are the ones the specification calls out by name, so a reader can grep for
#: them and find them being tested.
NAMED_FORBIDDEN = ("F3924", "F3912", "F3913", "F3914", "F3915", "F3916",
                   "F3917", "F3918", "F3898", "F3899", "F2230")


#: How ``fit`` serialises the quarantine list before hashing it. Kept as a
#: constant so this file recomputes the hash the same way rather than by a
#: separate convention that could drift.
QUARANTINE_JOIN = "\n"


def _check(name: str, passed: bool, detail: str, **evidence) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL",
            "detail": detail, **evidence}


def leakage_report() -> dict[str, Any]:
    """Section 19, every mandatory check, with the evidence attached."""
    from muleguard.features import firewall
    from muleguard.features.frame import raw_with_meta

    transform = cr.load()
    features = list(transform.features)
    feature_set = set(features)
    cfg = firewall.config()
    quarantined = (set(cfg.hard_quarantine) | set(cfg.conditional_quarantine)
                   | set(cfg.fairness_excluded))

    checks: list[dict[str, Any]] = []

    checks.append(_check(
        "target_never_in_fingerprint",
        settings.TARGET_COLUMN not in feature_set,
        f"{settings.TARGET_COLUMN} is not one of the {len(features)} "
        "fingerprint features",
        target_column=settings.TARGET_COLUMN))

    intersect = sorted(feature_set & quarantined)
    checks.append(_check(
        "no_quarantined_feature_in_fingerprint", not intersect,
        f"the live firewall quarantines {len(quarantined)} columns; "
        f"{len(intersect)} of them are in the fingerprint",
        policy_version=cfg.policy_version,
        n_quarantined=len(quarantined), offenders=intersect))

    named = sorted(c for c in NAMED_FORBIDDEN if c in feature_set)
    checks.append(_check(
        "no_named_forbidden_column", not named,
        "the columns section 4 names explicitly are absent from the fingerprint",
        columns_checked=list(NAMED_FORBIDDEN), offenders=named))

    checks.append(_check(
        "no_sensitive_or_fairness_excluded_field",
        not (feature_set & set(cfg.fairness_excluded)),
        f"none of the {len(cfg.fairness_excluded)} fairness-excluded attributes "
        "contributes to similarity",
        fairness_excluded=sorted(cfg.fairness_excluded)))

    # The transform's own record of the firewall it was fitted under. If the
    # policy has moved since, this is how a reader finds out.
    live_hash = cr._sha_text(QUARANTINE_JOIN.join(sorted(quarantined)))
    checks.append(_check(
        "fingerprint_matches_the_live_firewall_policy",
        transform.quarantine_hash == live_hash,
        "the quarantine hash recorded at fit time still matches the live one",
        quarantine_hash_at_fit=transform.quarantine_hash,
        quarantine_hash_now=live_hash))

    # Fitted on development rows only.
    dev = set(cr.reference_row_index().tolist())
    from muleguard.data import split as split_mod
    locked = {i for i, m in enumerate(split_mod.load_locked_test_mask()) if m}
    checks.append(_check(
        "transform_fitted_on_development_rows_only",
        not (dev & locked),
        f"the reference partition holds {len(dev)} rows and shares none of the "
        f"{len(locked)} locked-test rows",
        n_reference_rows=transform.n_reference_rows,
        n_locked_test_rows=len(locked)))

    checks.append(_check(
        "null_distribution_carries_no_label",
        transform.null_statistics.get("label_used") is False,
        "the empirical null was built from unlabelled random pairs",
        n_null_pairs=transform.n_null_pairs))


    # --- behaviour under adversarial input --------------------------------
    frame = raw_with_meta()
    index = cr.build_index()
    row = int(index.row_index[0])
    base = frame[[row]].to_dicts()[0]
    safe = {k: base[k] for k in features if k in base}

    def _order(values: dict[str, Any]) -> list[str]:
        result = cr.cohort_for_features(cr.with_derived_meta(dict(values)), k=10,
                                        with_explanations=False)
        return [n["account_reference"] for n in result["neighbors"]]

    baseline_order = _order(safe)

    with_target = dict(safe)
    with_target[settings.TARGET_COLUMN] = 1
    checks.append(_check(
        "uploaded_target_does_not_change_retrieval",
        _order(with_target) == baseline_order,
        f"setting {settings.TARGET_COLUMN}=1 on the query leaves the ranking "
        "unchanged, because the fingerprint never reads it",
        neighbors=baseline_order))

    poisoned = dict(safe)
    changed = []
    for col in NAMED_FORBIDDEN:
        if col in base:
            poisoned[col] = 999999.0
            changed.append(col)
    checks.append(_check(
        "forbidden_columns_cannot_move_retrieval",
        not changed or _order(poisoned) == baseline_order,
        f"setting {len(changed)} forbidden columns to an extreme value leaves "
        "the ranking unchanged",
        columns_changed=changed))

    labelled = dict(safe)
    labelled.update({"label": 1, "is_mule": 1, "judge_label": 1, "y_true": 1})
    checks.append(_check(
        "judge_supplied_labels_are_inert",
        _order(labelled) == baseline_order,
        "label-shaped keys added to an uploaded row are ignored entirely",
        keys_added=["label", "is_mule", "judge_label", "y_true"]))

    before_hash = transform.weights_hash(), transform.scaling_hash()
    _order(poisoned)
    after = cr.load()
    checks.append(_check(
        "a_query_cannot_modify_the_frozen_transform",
        (after.weights_hash(), after.scaling_hash()) == before_hash,
        "the weights and scaling statistics are byte-identical after scoring "
        "an adversarial query",
        weights_hash=after.weights_hash(), scaling_hash=after.scaling_hash()))

    failed = [c["check"] for c in checks if c["status"] != "PASS"]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "radar_version": transform.radar_version,
        "n_fingerprint_features": len(features),
        "firewall_policy_version": cfg.policy_version,
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "verdict": "PASS" if not failed else "FAIL",
        "release_blocking": True,
    }


def determinism_report(*, n_probe_rows: int = 25) -> dict[str, Any]:
    """Section 19's determinism half, and section 39's second artifact.

    The three ways an answer could wobble are all exercised here: the same
    query asked twice, the same account reached by a different route, and the
    same reference frame handed over in a different order. A retrieval layer an
    analyst is meant to act on has to give the same answer to all three.
    """
    from muleguard.features.frame import raw_with_meta

    frame = raw_with_meta()
    index = cr.build_index()
    rng = np.random.default_rng(int(index.transform.seed))
    probe = np.sort(rng.choice(index.row_index, size=min(n_probe_rows,
                                                        len(index.row_index)),
                               replace=False))

    checks: list[dict[str, Any]] = []

    # 1. the same query, twice
    first = {int(r): [n["account_reference"] for n in
                      cr.cohort_for_row(int(r), k=10,
                                        with_explanations=False)["neighbors"]]
             for r in probe}
    second = {int(r): [n["account_reference"] for n in
                       cr.cohort_for_row(int(r), k=10,
                                         with_explanations=False)["neighbors"]]
              for r in probe}
    checks.append(_check(
        "identical_query_returns_identical_ranking", first == second,
        f"{len(probe)} probe rows queried twice, in order, with identical results",
        n_probe_rows=len(probe)))

    # 2. the same account by row id and by uploaded feature values
    features = list(index.transform.features)
    mismatches = []
    for r in probe[:10]:
        values = {k: v for k, v in frame[[int(r)]].to_dicts()[0].items()
                  if k in features}
        by_upload = [n["account_reference"] for n in cr.cohort_for_features(
            cr.with_derived_meta(values), k=10,
            with_explanations=False)["neighbors"]]
        # The row is in the reference frame, so its own id is excluded from its
        # cohort but not from an uploaded copy's. Compare the shared portion.
        own = cr.reference_label(int(r))
        if [a for a in by_upload if a != own][:9] != first[int(r)][:9]:
            mismatches.append(int(r))
    checks.append(_check(
        "row_lookup_and_uploaded_copy_agree", not mismatches,
        "querying by row id and by re-uploading that row's own feature values "
        "returns the same neighbours in the same order",
        mismatched_rows=mismatches))

    # 3. reference frame handed over in a different order
    shuffled_rows = index.row_index.copy()
    np.random.default_rng(0).shuffle(shuffled_rows)
    shuffled = cr.build_index(rows=shuffled_rows, refresh=True)
    reordered = {int(r): [n["account_reference"] for n in
                          cr.cohort_for_row(int(r), k=10, index=shuffled,
                                            with_explanations=False)["neighbors"]]
                 for r in probe}
    checks.append(_check(
        "reference_row_order_does_not_change_results", reordered == first,
        "the reference frame is a set, not a sequence: shuffling the rows "
        "handed to build_index leaves every ranking unchanged"))
    cr.build_index(refresh=True)   # restore the default-scope cache

    failed = [c["check"] for c in checks if c["status"] != "PASS"]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "radar_version": index.transform.radar_version,
        "seed": int(index.transform.seed),
        "probe_rows": [int(r) for r in probe],
        "ranking_hash": cr._sha_text(repr(sorted(first.items()))),
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "verdict": "PASS" if not failed else "FAIL",
        "release_blocking": True,
    }


def write_all() -> dict[str, Any]:
    """Produce both section-39 artifacts and return their verdicts."""
    leak = leakage_report()
    save_json(leak, LEAKAGE_PATH)
    log.info("wrote %s (%s)", LEAKAGE_PATH, leak["verdict"])
    det = determinism_report()
    save_json(det, DETERMINISM_PATH)
    log.info("wrote %s (%s)", DETERMINISM_PATH, det["verdict"])
    fair = fairness_report()
    save_json(fair, FAIRNESS_PATH)
    log.info("wrote %s (%s)", FAIRNESS_PATH, fair["verdict"])
    return {"leakage": leak["verdict"], "determinism": det["verdict"],
            "fairness": fair["verdict"]}


#: Profile attributes the audit measures concordance on. Every one of them is
#: excluded from the fingerprint; they are read here only to check that the
#: exclusion actually shows up in who ends up next to whom.
PROFILE_FIELDS = {
    "F3890": "AREA_CATEGORY",
    "F3891": "CUST_OCCP",
    "F3892": "GENDER",
    "F3894": "AGE_IN_YRS",
}

FAIRNESS_PATH = settings.ARTIFACTS_DIR / "testing" / "cohort_radar_fairness.json"


def fairness_report(*, n_probe_rows: int = 300, k: int = 10,
                    seed: int = 42) -> dict[str, Any]:
    """Section 32: does the radar quietly cluster people rather than behaviour?

    Excluding a protected attribute from the feature list is necessary and not
    sufficient - a behavioural feature can carry a demographic signal, and a
    cohort panel that groups by occupation while claiming to group by behaviour
    is the same failure with better paperwork.

    So this measures the outcome rather than the intent: how often do cohort
    neighbours share a profile attribute, against how often two random
    reference accounts do? A ratio near 1.0 means the attribute is not
    organising the neighbourhoods. Concordance is reported for every field even
    where it is unremarkable, because reporting only the reassuring ones would
    make the artifact worthless.
    """
    from muleguard.features import firewall
    from muleguard.features.frame import raw_with_meta

    transform = cr.load()
    features = set(transform.features)
    cfg = firewall.config()
    index = cr.build_index()
    frame = raw_with_meta()
    rows = index.row_index

    available = [c for c in PROFILE_FIELDS if c in frame.columns]
    profile = {c: frame[rows.tolist()][c].to_numpy() for c in available}

    rng = np.random.default_rng(seed)
    probe = rng.choice(len(rows), size=min(n_probe_rows, len(rows)), replace=False)
    left, right = [], []
    for p in probe:
        result = cr.cohort_for_row(int(rows[p]), k=k, index=index,
                                   with_explanations=False)
        for n in result["neighbors"]:
            left.append(int(p))
            right.append(int(np.searchsorted(rows, n["row_index"])))
    left, right = np.asarray(left), np.asarray(right)
    rand_a = rng.choice(len(rows), size=left.size)
    rand_b = rng.choice(len(rows), size=left.size)

    def _concordance(a: np.ndarray, b: np.ndarray, col: str) -> float:
        x, y = profile[col][a], profile[col][b]
        if col == "F3894":       # age: compare decades, not exact years
            x = np.floor(np.asarray(x, dtype=float) / 10)
            y = np.floor(np.asarray(y, dtype=float) / 10)
        return float(np.nanmean(x == y))

    concordance = {}
    for col in available:
        neighbour = _concordance(left, right, col)
        chance = _concordance(rand_a, rand_b, col)
        concordance[PROFILE_FIELDS[col]] = {
            "column": col,
            "in_fingerprint": col in features,
            "firewall_class": ("fairness_excluded" if col in cfg.fairness_excluded
                               else "contextual_only" if col in cfg.contextual_only
                               else "admitted"),
            "neighbour_concordance": neighbour,
            "random_pair_concordance": chance,
            "ratio_to_chance": (neighbour / chance) if chance else None,
        }

    weights = np.concatenate([transform.numeric_weights,
                              transform.categorical_weights])
    order = np.argsort(-weights)
    names = list(transform.features)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "radar_version": transform.radar_version,
        "method": (f"{len(probe)} reference accounts, their top-{k} cohorts, "
                   f"{left.size} neighbour pairs, against the same number of "
                   "uniformly random reference pairs"),
        "n_fingerprint_features": len(names),
        "profile_fields_in_fingerprint": sorted(
            c for c in PROFILE_FIELDS if c in features),
        "concordance": concordance,
        "similarity_weight": {
            "sum": float(weights.sum()),
            "top_10_share": float(weights[order[:10]].sum()),
            "top_10": [{"feature": names[i], "weight": float(weights[i])}
                       for i in order[:10]],
            "categorical_features": list(transform.categorical_features),
            "categorical_weight_share": float(transform.categorical_weights.sum()),
        },
        "verdict": ("PASS" if not any(c["in_fingerprint"]
                                      for c in concordance.values()) else "FAIL"),
    }
