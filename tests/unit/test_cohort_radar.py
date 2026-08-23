"""Cohort Radar: retrieval that must never become a second classifier.

The tests are organised around the two ways this layer could do damage. It
could *see* something the firewall excluded - the target, a post-resolution
field, a fairness-excluded attribute - and launder it back into the product as
a similarity. Or it could *say* something the data cannot support: that two
accounts share a handler, a network, a controller. Sections 13, 19 and 37 exist
because both failures look like a working feature from the outside.

Everything below is retrieval-side. Not one assertion here touches a
probability, a tier or a threshold, and that is the point.
"""
from __future__ import annotations

import numpy as np
import pytest

from muleguard import settings
from muleguard.usp import cohort_radar as cr

TRANSFORM = cr.TRANSFORM_PATH
BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
needs_transform = pytest.mark.skipif(
    not TRANSFORM.exists(),
    reason="cohort transform not built - python -m muleguard.cli.build_cohort_radar")
needs_index = pytest.mark.skipif(
    not (TRANSFORM.exists() and BUNDLE.exists()),
    reason="cohort transform or champion bundle not built")


# --------------------------------------------------------------------------
# section 5 / 13 - the language guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("claim", [
    "these accounts are in the same criminal network",
    "operated by the same mule handler",
    "a connected mule ring",
    "members of the same syndicate",
    "controlled by same person",
])
def test_forbidden_claims_are_refused(claim):
    with pytest.raises(ValueError, match="behaviourally similar"):
        cr.assert_language_safe({"finding": claim})


@pytest.mark.parametrize("relation", ["SENT_MONEY_TO", "CONTROLLED_BY", "SAME_HANDLER"])
def test_forbidden_edge_labels_are_refused_however_spelled(relation):
    """An enum is a claim too. Separator normalisation is what makes it one."""
    with pytest.raises(ValueError):
        cr.assert_language_safe({"edges": [{"relation": relation}]})
    with pytest.raises(ValueError):
        cr.assert_language_safe({"edges": [{"relation": relation.lower().replace("_", " ")}]})


def test_guard_reads_keys_not_only_values():
    with pytest.raises(ValueError):
        cr.assert_language_safe({"same_criminal_network": ["RV-1"]})


def test_disclaimer_may_name_what_it_denies():
    """A guard that cannot tell "we never say X" from "X" would force the
    product to stop explaining itself in order to stay compliant."""
    cr.assert_language_safe({
        "disclaimer": cr.DISCLAIMER,
        "interpretation": cr.INTERPRETATION,
        "action_policy": cr.ACTION_POLICY,
        "language_guard": {"never_emitted": list(cr.FORBIDDEN_COHORT_LANGUAGE)},
    })


def test_the_only_edge_label_is_the_behavioural_one():
    assert cr.EDGE_LABEL == "BEHAVIORALLY_SIMILAR_TO"
    cr.assert_language_safe({"relation": cr.EDGE_LABEL})


def test_disclaimer_states_the_limitation_in_product_language():
    text = cr.DISCLAIMER.lower()
    assert "behavioural similarity does not establish" in text
    assert "prioritisation signal only" in text
    assert "behaviourally similar accounts" in cr.INTERPRETATION.lower()


# --------------------------------------------------------------------------
# section 19 - what the fingerprint may see
# --------------------------------------------------------------------------


@needs_index
def test_target_is_not_in_the_fingerprint():
    assert "F3924" not in set(cr.fingerprint_features())


@needs_index
@pytest.mark.parametrize("col", ["F3924", "F3912", "F3913", "F3914", "F3915",
                                 "F3916", "F3917", "F3918", "F3898", "F3899",
                                 "F2230"])
def test_quarantined_and_post_outcome_columns_are_absent(col):
    assert col not in set(cr.fingerprint_features())


@needs_index
def test_fingerprint_passes_the_live_firewall_not_a_frozen_copy():
    """The quarantine is a live config. A feature reclassified tomorrow must
    break the radar rather than quietly remain inside a frozen fingerprint."""
    from muleguard.features import firewall
    from muleguard.features.frame import augmented_registry

    firewall.assert_clean(cr.fingerprint_features(),
                          context="test_cohort_fingerprint",
                          registry=augmented_registry())


@needs_transform
def test_transform_was_fitted_on_the_development_partition_only():
    from muleguard.data import split as split_mod

    rows = cr.reference_row_index()
    locked = np.flatnonzero(np.asarray(split_mod.load_locked_test_mask()))
    assert np.intersect1d(rows, locked).size == 0


@needs_transform
def test_weights_are_a_distribution_over_the_fingerprint():
    t = cr.load()
    w = np.concatenate([t.numeric_weights, t.categorical_weights])
    assert len(w) == len(t.features)
    assert w.min() >= 0.0
    assert abs(float(w.sum()) - 1.0) < 1e-9


@needs_transform
def test_bands_are_percentiles_of_an_empirical_null_and_ordered():
    t = cr.load()
    assert t.band_percentiles[cr.BAND_VERY_HIGH] == 99.5
    assert t.band_percentiles[cr.BAND_HIGH] == 99.0
    assert t.band_percentiles[cr.BAND_MODERATE] == 95.0
    assert (t.bands[cr.BAND_VERY_HIGH] >= t.bands[cr.BAND_HIGH]
            >= t.bands[cr.BAND_MODERATE])


@needs_transform
def test_null_was_sampled_without_reading_a_label():
    assert cr.load().null_statistics["label_used"] is False


# --------------------------------------------------------------------------
# section 37 cases 1-6, 9-11, 15 - what must not move the neighbours
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def index():
    return cr.build_index()


@pytest.fixture(scope="module")
def query_row(index):
    """A real development row, as raw values, plus its reference-frame position."""
    from muleguard.features.frame import raw_with_meta

    position = int(np.argmax(index.risk))
    row = int(index.row_index[position])
    return {"row": row, "position": position,
            "values": raw_with_meta()[row].to_dicts()[0]}


def _order(payload) -> list[str]:
    return [n["account_reference"] for n in payload["neighbors"]]


def _sims(payload) -> list[float]:
    return [n["behavioral_similarity"] for n in payload["neighbors"]]


@needs_index
def test_case_1_same_row_twice_gives_identical_ordering(index, query_row):
    a = cr.cohort_for_row(query_row["row"], index=index, k=10)
    b = cr.cohort_for_row(query_row["row"], index=index, k=10)
    assert _order(a) == _order(b)
    assert _sims(a) == _sims(b)


@needs_index
def test_case_2_reference_row_order_does_not_change_the_neighbours(index, query_row):
    """Shuffling the reference frame must not reorder retrieval."""
    rng = np.random.default_rng(11)
    shuffled = index.row_index.copy()
    rng.shuffle(shuffled)
    other = cr.build_index(rows=shuffled)
    assert _order(cr.cohort_for_row(query_row["row"], index=index, k=10)) == \
           _order(cr.cohort_for_row(query_row["row"], index=other, k=10))


    # Second half of the guarantee: even if an unsorted frame did reach
    # retrieval, ties break on row index rather than on array position.
    perm = rng.permutation(len(index.row_index))
    permuted = cr.CohortIndex(
        transform=index.transform, row_index=index.row_index[perm],
        references=[index.references[i] for i in perm],
        numeric=index.numeric[perm], categorical=index.categorical[perm],
        risk=index.risk[perm], tier=[index.tier[i] for i in perm],
        patterns=[index.patterns[i] for i in perm], scope=index.scope)
    straight = cr.find_neighbors(index=index, q_num=index.numeric[0],
                                 q_cat=index.categorical[0], k=10,
                                 with_explanations=False)
    scrambled = cr.find_neighbors(index=permuted, q_num=index.numeric[0],
                                  q_cat=index.categorical[0], k=10,
                                  with_explanations=False)
    assert [n["account_reference"] for n in straight] ==            [n["account_reference"] for n in scrambled]


@needs_index
@pytest.mark.parametrize("col,value", [
    ("F3924", 1), ("F3924", 0),            # case 3: the target
    ("F3912", 999.0), ("F3918", -999.0),   # case 4: quarantined columns
    ("F2230", 12345.0),                    # case 5
    ("F3898", 1.0), ("F3899", 1.0),        # case 6: post-resolution fields
])
def test_cases_3_to_6_forbidden_columns_cannot_move_retrieval(index, query_row,
                                                              col, value):
    base = cr.cohort_for_features(query_row["values"], index=index, k=10)
    tampered = dict(query_row["values"])
    tampered[col] = value
    after = cr.cohort_for_features(tampered, index=index, k=10)
    assert _order(base) == _order(after)
    assert _sims(base) == _sims(after)


@needs_index
def test_case_7_a_material_change_to_a_safe_feature_moves_similarity(index, query_row):
    """The mirror of the tests above: if nothing moved retrieval, retrieval
    would be measuring nothing."""
    t = index.transform
    j = int(np.argmax(t.numeric_weights))
    heaviest = t.numeric_features[j]
    base = cr.cohort_for_features(query_row["values"], index=index, k=10)
    moved = dict(query_row["values"])
    current = moved.get(heaviest)
    moved[heaviest] = (float(current) + 50.0 * float(t.scale[j])
                       if current is not None else 1e6)
    after = cr.cohort_for_features(moved, index=index, k=10)
    assert (after["neighbors"][0]["behavioral_similarity"]
            < base["neighbors"][0]["behavioral_similarity"])


@needs_index
def test_case_8_an_empty_query_is_answered_as_insufficient_data(index):
    payload = cr.cohort_for_features({}, index=index, k=5)
    assert payload["cohort_summary"]["insufficient_data"] is True
    assert payload["neighbors"] == []
    assert payload["mutual_edges"] == []
    assert payload["data_sufficiency"]["status"] == "INSUFFICIENT_DATA"
    # The refusal still carries the framing; a bare error reads as a fault.
    assert payload["disclaimer"] == cr.DISCLAIMER


@needs_index
def test_a_populated_cohort_is_not_flagged_insufficient(index, query_row):
    payload = cr.cohort_for_row(query_row["row"], index=index, k=5)
    assert payload["cohort_summary"]["insufficient_data"] is False
    assert "data_sufficiency" not in payload


@needs_transform
def test_the_sufficiency_floor_is_measured_not_chosen():
    """Every reference account clears its own floor by construction. That is
    what makes it a floor rather than a number somebody picked."""
    t = cr.load()
    assert 0.0 < t.min_reference_coverage <= 1.0
    assert t.min_reference_coverage <= t.null_statistics["median_reference_coverage"]


@needs_index
def test_case_9_the_query_row_is_never_its_own_neighbour(index, query_row):
    payload = cr.cohort_for_row(query_row["row"], index=index, k=10)
    assert cr.reference_label(query_row["row"]) not in _order(payload)


@needs_index
def test_case_10_duplicate_rows_are_handled_deterministically(index, query_row):
    """Ties break on row_index, not on whatever order the sort happened to see."""
    rows = np.sort(np.concatenate([index.row_index, index.row_index[:3]]))
    a = cr.build_index(rows=rows)
    b = cr.build_index(rows=rows[::-1].copy())
    assert _order(cr.cohort_for_features(query_row["values"], index=a, k=10)) == \
           _order(cr.cohort_for_features(query_row["values"], index=b, k=10))


@needs_index
def test_case_11_an_unseen_category_is_handled_not_crashed(index, query_row):
    t = index.transform
    if not t.categorical_features:
        pytest.skip("fingerprint has no categorical feature")
    odd = dict(query_row["values"])
    odd[t.categorical_features[0]] = "A-CATEGORY-THE-TRAINING-DATA-NEVER-SAW"
    assert len(cr.cohort_for_features(odd, index=index, k=5)["neighbors"]) == 5


@needs_index
def test_case_15_labels_supplied_with_a_query_do_not_reach_the_radar(index, query_row):
    """A judge upload arrives with a target column. The radar must ignore it,
    and the way to be sure is that the answer is identical without it."""
    labelled = dict(query_row["values"])
    labelled.update({"F3924": 1, "label": 1, "is_mule": "yes"})
    unlabelled = {k: v for k, v in query_row["values"].items()
                  if k not in {"F3924", "label", "is_mule"}}
    assert _order(cr.cohort_for_features(labelled, index=index, k=10)) == \
           _order(cr.cohort_for_features(unlabelled, index=index, k=10))


@needs_index
def test_a_query_cannot_modify_the_frozen_transform(index, query_row):
    t = index.transform
    before = (t.median.copy(), t.scale.copy(), t.numeric_weights.copy())
    extreme = dict(query_row["values"])
    for name in t.numeric_features[:20]:
        extreme[name] = 1e12
    cr.cohort_for_features(extreme, index=index, k=5)
    assert np.array_equal(before[0], t.median)
    assert np.array_equal(before[1], t.scale)
    assert np.array_equal(before[2], t.numeric_weights)


# --------------------------------------------------------------------------
# sections 11-14 - what the output is allowed to say
# --------------------------------------------------------------------------


@needs_index
def test_pattern_agreement_is_reported_separately_never_blended(index, query_row):
    payload = cr.cohort_for_row(query_row["row"], index=index, k=5)
    for n in payload["neighbors"]:
        assert "behavioral_similarity" in n and "pattern_similarity" in n
        # There is no blend, so there is no key that could carry one.
        assert not ({"final_risk", "combined_score", "blended_score"} & set(n))


@needs_index
def test_every_neighbour_carries_its_own_unmodified_tier(index, query_row):
    payload = cr.cohort_for_row(query_row["row"], index=index, k=5)
    for n in payload["neighbors"]:
        pos = index.position_of(n["account_reference"])
        assert n["neighbor_risk_probability"] == float(index.risk[pos])
        assert n["neighbor_risk_tier"] == index.tier[pos]


@needs_index
def test_mutual_edges_require_agreement_in_both_directions(index, query_row):
    payload = cr.cohort_for_row(query_row["row"], index=index, k=10)
    shown = set(_order(payload))
    for e in payload["mutual_edges"]:
        assert e["relationship"] == cr.EDGE_LABEL
        assert e["target"] in shown
        assert e["mutual"] is True


@needs_index
def test_similarity_is_symmetric(index):
    """Not decoration: an asymmetric similarity would make a mutual edge mean
    something different in each direction."""
    t = index.transform
    ab = t.similarity(index.numeric[0], index.categorical[0],
                      index.numeric[1:2], index.categorical[1:2])
    ba = t.similarity(index.numeric[1], index.categorical[1],
                      index.numeric[0:1], index.categorical[0:1])
    assert float(ab[0]) == pytest.approx(float(ba[0]), abs=1e-15)


@needs_index
def test_self_similarity_is_exactly_one(index):
    s = index.transform.similarity(index.numeric[0], index.categorical[0],
                                   index.numeric[:1], index.categorical[:1])
    assert float(s[0]) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# helpers that carry real weight
# --------------------------------------------------------------------------


def test_jaccard_of_two_empty_pattern_sets_is_zero_not_one():
    """No shared patterns because neither account matched any is agreement
    about nothing. Reporting 1.0 there would invent evidence."""
    assert cr.jaccard(frozenset(), frozenset()) == 0.0


def test_jaccard_is_the_ordinary_definition():
    assert cr.jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)
    assert cr.jaccard(frozenset({"a"}), frozenset({"a"})) == 1.0


def test_reference_labels_round_trip_and_reject_foreign_references():
    assert cr.reference_label(42) == "RV-42"
    assert cr.row_index_for_reference("ACCT-42") is None
    assert cr.row_index_for_reference("") is None
    assert cr.row_index_for_reference("RV-not-a-number") is None


@needs_transform
def test_a_locked_test_row_does_not_resolve_to_the_reference_frame():
    from muleguard.data import split as split_mod

    locked = np.flatnonzero(np.asarray(split_mod.load_locked_test_mask()))
    assert cr.row_index_for_reference(f"RV-{int(locked[0])}") is None


def test_derived_meta_fills_gaps_and_never_overwrites_what_was_supplied():
    """A partial row derives MG_* wrongly rather than loudly. Supplied values
    win, or a caller who sent 120 model features would have them silently
    recomputed as null."""
    out = cr.with_derived_meta({"MG_PASSTHROUGH_7D": 100.0,
                                "MG_RAIL_FRAGMENTATION": 2.0})
    assert out["MG_PASSTHROUGH_7D"] == 100.0
    assert out["MG_RAIL_FRAGMENTATION"] == 2.0


def test_derived_meta_survives_a_row_it_cannot_derive_from():
    assert cr.with_derived_meta({"F1": 1.0})["F1"] == 1.0


def test_pattern_feature_names_match_the_card_definitions():
    from muleguard.explain.pattern_cards import PATTERN_DEFINITIONS

    assert set(cr.pattern_feature_names()) == {d.feature for d in PATTERN_DEFINITIONS}


def test_experimental_classifier_integration_is_off():
    """Section 27. The path does not exist; this is the switch that would have
    to be flipped before it could."""
    assert settings.load_config("cohort_radar")["EXPERIMENTAL_COHORT_FEATURES"] is False


@needs_transform
def test_manifest_records_every_provenance_field_section_39_requires():
    m = cr.manifest()
    required = {
        "source data hash": m["source_data"]["dataset_sha256"],
        "feature list": m["fingerprint"]["features"],
        "quarantine list hash": m["quarantine"]["quarantine_hash"],
        "scaling statistics hash": m["scaling"]["statistics_hash"],
        "weight source": m["weights"]["primary"],
        "similarity formula": m["similarity_formula"],
        "percentile thresholds": m["percentile_thresholds"]["bands"],
        "build timestamp": m["generated_utc"],
        "git SHA": m["git"]["commit_sha"],
    }
    for label, value in required.items():
        assert value, f"manifest is missing {label}"
    assert m["quarantine"]["target_in_fingerprint"] is False
    assert m["affects_model_score"] is False
    assert m["edge_relationship"] == cr.EDGE_LABEL
