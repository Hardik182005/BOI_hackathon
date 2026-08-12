"""Missed-Mule Error Atlas: classifier invariants and output safety.

The Atlas is a diagnostic instrument. Two properties matter more than any
number it produces:

* the classification is deterministic and identical for every account - if it
  could vary, or could be steered per account, it would be a patch mechanism
  wearing a diagnostic's clothes;
* nothing it emits can name a quarantined column or use accusatory vocabulary.

Both are tested here against the engine directly, and against the generated
artifact when one is present.
"""
import json
import random

import numpy as np
import pytest

from muleguard import settings
from muleguard.models import error_atlas as atlas

ARTIFACT = settings.METRICS_DIR / "error_atlas.json"
DOC = settings.DOCS_DIR / "ERROR_ATLAS.md"

# Assembled from fragments so the literals never appear in this file: the
# release gate scans the repository for these tokens and a test that hard-coded
# them would either trip the gate or force an exemption that weakens it.
FORBIDDEN_WORDS = (
    "GUI" + "LTY",
    "CRIM" + "INAL",
    "PERMANENTLY" + "_SAFE",
    "CERTIFIED" + "_CLEAN",
    "AUTO" + "_FREEZE",
)


def _m(**over):
    """A benign measurement vector: nothing fires except the residual."""
    base = dict(
        champion_rank=5000, budget_k=100,
        missing_fraction=0.05, missingness_z=0.1,
        range_violation_share=0.0,
        knn_distance=10.0, knn_threshold=1e6,
        n_peer_families_within_budget=0,
        n_mule_labels_in_neighbourhood=3,
        distance_to_nearest_known_mule=1.0,
        distance_to_nearest_legitimate_account=2.0,
        n_families_below_dev_median=0, n_families_measured=4,
    )
    base.update(over)
    return atlas.MissMeasurements(**base)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_classifier_is_deterministic_over_many_random_measurements():
    rng = random.Random(20260812)
    for _ in range(300):
        m = _m(
            champion_rank=rng.randint(1, 7264),
            missing_fraction=rng.random(),
            missingness_z=rng.uniform(-2, 8),
            range_violation_share=rng.random(),
            knn_distance=rng.uniform(0, 100),
            knn_threshold=rng.choice([50.0, 1e6]),
            n_peer_families_within_budget=rng.randint(0, 3),
            n_mule_labels_in_neighbourhood=rng.randint(0, 10),
            distance_to_nearest_known_mule=rng.uniform(0.1, 50),
            distance_to_nearest_legitimate_account=rng.uniform(0.1, 50),
        )
        first = atlas.classify(m)
        for _ in range(5):
            assert atlas.classify(m) == first
        assert first["category"] in atlas.CATEGORIES


def test_classification_does_not_depend_on_evaluation_order_of_other_rows():
    """Each account is classified in isolation - no cross-row state."""
    rows = [_m(champion_rank=r, n_peer_families_within_budget=r % 3)
            for r in (150, 400, 900, 3000)]
    forward = [atlas.classify(m)["category"] for m in rows]
    backward = [atlas.classify(m)["category"] for m in reversed(rows)][::-1]
    assert forward == backward


def test_classifier_holds_no_state_between_calls():
    m = _m(missingness_z=9.0)
    a = atlas.classify(m)
    atlas.classify(_m(champion_rank=1))
    atlas.classify(_m(n_mule_labels_in_neighbourhood=0,
                      distance_to_nearest_legitimate_account=0.1))
    assert atlas.classify(m) == a


def test_rule_constants_are_frozen():
    with pytest.raises(Exception):
        atlas.DEFAULT_CONSTANTS.missingness_z = 1.0  # type: ignore[misc]


def test_measurements_are_frozen():
    with pytest.raises(Exception):
        _m().champion_rank = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The rule list itself
# ---------------------------------------------------------------------------


def test_every_category_is_reachable():
    got = {
        "MISSING_DATA": atlas.classify(_m(missingness_z=5.0)),
        "OOD_PATTERN": atlas.classify(_m(range_violation_share=0.9)),
        "THRESHOLD_MISS": atlas.classify(_m(champion_rank=150)),
        "MODEL_DISAGREEMENT": atlas.classify(_m(n_peer_families_within_budget=2)),
        "LOOKALIKE_MULE": atlas.classify(
            _m(n_mule_labels_in_neighbourhood=0,
               distance_to_nearest_legitimate_account=0.5,
               distance_to_nearest_known_mule=9.0)),
        "LOW_SIGNAL_MULE": atlas.classify(_m()),
    }
    for expected, verdict in got.items():
        assert verdict["category"] == expected
    assert set(got) == set(atlas.CATEGORIES)


def test_first_matching_rule_wins_in_documented_order():
    """A row that satisfies several rules takes the highest-priority one."""
    m = _m(missingness_z=9.0, range_violation_share=0.9, champion_rank=10,
           n_peer_families_within_budget=3,
           n_mule_labels_in_neighbourhood=0,
           distance_to_nearest_legitimate_account=0.1,
           distance_to_nearest_known_mule=9.0)
    v = atlas.classify(m)
    assert v["category"] == "MISSING_DATA"
    assert v["decided_by_rule"] == "missing_data"
    # the losing explanations are still recorded rather than hidden
    assert {"ood_pattern", "threshold_miss", "model_disagreement",
            "lookalike_mule"} <= set(v["other_rules_that_also_fired"])


def test_every_rule_is_evaluated_and_reported_for_every_row():
    v = atlas.classify(_m())
    assert [e["rule"] for e in v["rules_evaluated"]] == list(atlas.RULE_ORDER)
    assert all("test" in e and "measured" in e for e in v["rules_evaluated"])


def test_rule_thresholds_are_boundary_exact():
    c = atlas.DEFAULT_CONSTANTS
    assert atlas.classify(_m(missingness_z=c.missingness_z))["category"] == "MISSING_DATA"
    assert atlas.classify(_m(missingness_z=c.missingness_z - 1e-9))["category"] != "MISSING_DATA"
    k = 100
    assert atlas.classify(_m(champion_rank=200, budget_k=k))["category"] == "THRESHOLD_MISS"
    assert atlas.classify(_m(champion_rank=201, budget_k=k))["category"] != "THRESHOLD_MISS"


def test_lookalike_needs_both_conditions():
    # a mule-labelled neighbour present -> not a lookalike
    assert atlas.classify(_m(n_mule_labels_in_neighbourhood=1,
                             distance_to_nearest_legitimate_account=0.1,
                             distance_to_nearest_known_mule=9.0)
                          )["category"] == "LOW_SIGNAL_MULE"
    # nearest mule-labelled account is closer -> not a lookalike
    assert atlas.classify(_m(n_mule_labels_in_neighbourhood=0,
                             distance_to_nearest_legitimate_account=9.0,
                             distance_to_nearest_known_mule=0.1)
                          )["category"] == "LOW_SIGNAL_MULE"


def test_rule_book_substitutes_the_constants_and_covers_every_category():
    book = atlas.rule_book()
    assert [d["category"] for d in book] == [
        "MISSING_DATA", "OOD_PATTERN", "THRESHOLD_MISS", "MODEL_DISAGREEMENT",
        "LOOKALIKE_MULE", "LOW_SIGNAL_MULE"]
    assert "{" not in "".join(d["test"] for d in book)
    assert "4.0" in book[0]["test"]


def test_firing_rates_over_a_control_group():
    rates = atlas.rule_firing_rates([_m(), _m(champion_rank=10)])
    assert rates["n_control_rows"] == 2
    assert rates["rates"]["threshold_miss"]["n_fired"] == 1
    assert rates["rates"]["threshold_miss"]["rate"] == 0.5
    assert "low_signal_mule" not in rates["rates"]  # residual has no base rate


# ---------------------------------------------------------------------------
# Quarantine guard
# ---------------------------------------------------------------------------


def test_quarantined_feature_anywhere_in_a_payload_is_refused():
    for payload in (
        {"top_shap_features": [{"feature": "F3898"}]},
        {"a": {"b": ["F2230"]}},
        {"missing_features": ["F0001", "F3924"]},
        {"F3912": 1.0},                       # as a key, not only a value
        ["fine", "the value of F3899 was high"],  # embedded in prose
    ):
        with pytest.raises(ValueError, match="quarantined"):
            atlas.assert_no_quarantined_feature(payload)


def test_clean_payload_passes_and_similar_names_are_not_false_positives():
    atlas.assert_no_quarantined_feature({
        "features": ["F0193", "F3908", "F39240", "F223", "MG_missing_share"],
        "note": "13 columns are quarantined and none are named here",
    })


def test_target_column_is_treated_as_quarantined():
    assert settings.TARGET_COLUMN in atlas.QUARANTINED_FEATURES


def test_top_attributions_never_emits_a_quarantined_feature():
    names = ["F0001", "F3898", "F0002"]          # middle one is quarantined
    contrib = np.array([[0.1, 9.9, 0.2], [0.1, 9.9, 0.2]])
    out = atlas.top_attributions(contrib, names, np.array([1.0, 2.0, 3.0]))
    assert [d["feature"] for d in out] == ["F0002", "F0001"]
    atlas.assert_no_quarantined_feature(out)


# ---------------------------------------------------------------------------
# Read-only contract
# ---------------------------------------------------------------------------


def test_read_only_contract_detects_a_changed_input():
    before = {"oof": "aaa", "bundle": "bbb"}
    atlas.assert_read_only_contract(before, dict(before))
    with pytest.raises(RuntimeError, match="read-only"):
        atlas.assert_read_only_contract(before, {"oof": "aaa", "bundle": "ccc"})


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def test_dense_rank_desc():
    r = atlas.dense_rank_desc(np.array([0.1, 0.9, 0.5]))
    assert r.tolist() == [3, 1, 2]


def test_robust_scale_matches_the_lens_convention():
    X = np.array([[1.0, np.nan]])
    Z = atlas.robust_scale(X, np.array([1.0, 4.0]), np.array([2.0, 2.0]))
    assert Z.tolist() == [[0.0, 0.0]]  # missing values land on the median


def test_neighbour_geometry_excludes_the_account_itself():
    Z = np.array([[0.0], [0.1], [5.0], [5.1], [9.0]])
    y = np.array([1, 0, 1, 0, 0])
    g = atlas.neighbour_geometry(Z, y, [0], k=3)[0]
    assert g["nearest_mule_labelled_row"] == 2          # not itself
    assert g["nearest_legitimate_row"] == 1
    assert g["distance_to_nearest_known_mule"] == pytest.approx(5.0)
    assert g["distance_to_nearest_legitimate_account"] == pytest.approx(0.1)
    assert 0 not in g["neighbourhood_rows"]


def test_percentile_of():
    ref = np.arange(100.0)
    assert atlas.percentile_of(-1, ref) == 0.0
    assert atlas.percentile_of(200, ref) == 100.0


def test_missingness_profile_groups_by_feature_family():
    prof = atlas.missingness_profile(
        np.array([np.nan, 1.0, np.nan]), ["A", "B", "C"],
        {"A": "CASH", "B": "CASH", "C": "GST"}, 0.1, 0.05)
    assert prof["n_missing"] == 2
    assert prof["missing_fraction"] == pytest.approx(2 / 3)
    assert prof["missing_by_feature_family"] == {"CASH": 1, "GST": 1}
    # reported to 4 decimals: this field is for the reader, while the classifier
    # reads the unrounded value measured in the CLI
    assert prof["missingness_z_vs_development"] == pytest.approx(
        (2 / 3 - 0.1) / 0.05, abs=1e-4)


def test_attribution_reports_unstable_signs_rather_than_hiding_them():
    contrib = np.array([[1.0, 0.5], [-1.0, 0.5]])   # first flips between repeats
    out = {d["feature"]: d for d in atlas.top_attributions(
        contrib, ["A", "B"], np.array([1.0, 2.0]))}
    assert out["A"]["stable_sign_across_repeats"] is False
    assert out["B"]["stable_sign_across_repeats"] is True


# ---------------------------------------------------------------------------
# The generated artifact
# ---------------------------------------------------------------------------


needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="run `python -m muleguard.cli.error_atlas` first")


@pytest.fixture(scope="module")
def payload():
    if not ARTIFACT.exists():
        pytest.skip("artifact not built")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@needs_artifact
def test_artifact_names_no_quarantined_column(payload):
    atlas.assert_no_quarantined_feature(payload)


@needs_artifact
def test_artifact_and_doc_use_no_forbidden_vocabulary(payload):
    blobs = [json.dumps(payload)]
    if DOC.exists():
        blobs.append(DOC.read_text(encoding="utf-8"))
    for blob in blobs:
        for word in FORBIDDEN_WORDS:
            assert word not in blob


@needs_artifact
def test_every_miss_carries_every_required_field(payload):
    required = ("closest_known_mule", "closest_legitimate_account",
                "family_scores", "top_shap_features", "missingness_pattern",
                "merchant_context_evidence", "anomaly_score",
                "nearest_neighbour_distance", "category", "classification",
                "measurements")
    assert payload["misses"], "an atlas with no misses proves nothing"
    for m in payload["misses"]:
        for field in required:
            assert field in m, f"row {m['dataset_row_index']} missing {field}"
        assert set(m["family_scores"]) == {"catboost", "lightgbm", "tabpfn",
                                           "xgboost"}
        assert m["category"] in atlas.CATEGORIES


@needs_artifact
def test_artifact_classification_reproduces_from_its_own_recorded_measurements(payload):
    """The stored category must follow from the stored numbers, not from prose."""
    for m in payload["misses"]:
        again = atlas.classify(atlas.MissMeasurements(**m["measurements"]))
        assert again["category"] == m["category"]
        assert again["decided_by_rule"] == m["classification"]["decided_by_rule"]


@needs_artifact
def test_category_counts_add_up(payload):
    counts = payload["categories"]["counts"]
    assert sum(counts.values()) == payload["scope"]["n_missed"]
    assert sum(counts.values()) == len(payload["misses"])


@needs_artifact
def test_neighbours_are_labelled_as_feature_space_only(payload):
    for m in payload["misses"]:
        for key in ("closest_known_mule", "closest_legitimate_account"):
            assert m[key]["relation"] == atlas.NEIGHBOUR_RELATION
            assert m[key]["not_a_transaction_link"] is True


@needs_artifact
def test_artifact_declares_the_read_only_and_no_llm_contracts(payload):
    ro = payload["read_only_contract"]
    assert ro["modifies_scores_thresholds_features_or_models"] is False
    assert set(ro["writes"]) == {"artifacts/metrics/error_atlas.json",
                                 "docs/ERROR_ATLAS.md"}
    assert payload["provenance"]["no_language_model_used"] is True


@needs_artifact
def test_attribution_method_is_stated_not_substituted(payload):
    am = payload["attribution_method"]
    assert am["method"] in ("EXACT_TREESHAP", "MODEL_FEATURE_IMPORTANCE",
                            "NOT_COMPUTED")
    if am["method"] == "EXACT_TREESHAP":
        assert "attribution_fidelity" in am
        assert am["fold_checks"], "fidelity claimed without per-fold evidence"


@needs_artifact
def test_hypotheses_are_marked_untested_and_unacted_on(payload):
    hy = payload["hypotheses_for_nested_cv_testing"]
    assert "nested cross-validation" in hy["standing_rule"]
    for h in hy["hypotheses"]:
        assert h["status"].startswith(("UNTESTED", "OBSERVATION"))
        assert h["trigger"] and h["how_to_test"]
