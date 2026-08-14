"""Pattern cards describe measured behaviour and nothing else.

The risk here is scope creep: a card layer that starts matching typologies the
dataset cannot support turns the ProofGraph into a second, unvalidated model
wearing the first one's credibility.
"""
from __future__ import annotations

from muleguard.explain.pattern_cards import (
    PATTERN_DEFINITIONS,
    match_patterns,
)


def test_no_patterns_match_an_empty_row():
    assert match_patterns({}) == []


def test_a_quiet_account_matches_nothing():
    quiet = {"MG_PASSTHROUGH_7D": 0.1, "MG_RETENTION_RATIO": 0.9,
             "MG_BURST_7_31": 0.0, "MG_CASHOUT_PRESSURE": 0.0,
             "MG_BALANCE_DRAIN": 0.2, "MG_ALERT_CONVERGENCE": 0.0}
    assert match_patterns(quiet) == []


def test_passthrough_matches_and_reports_its_evidence():
    cards = match_patterns({"MG_PASSTHROUGH_7D": 1.4})
    assert len(cards) == 1
    card = cards[0]
    assert card["id"] == "passthrough_7d"
    assert card["evidence"]["observed_value"] == 1.4
    assert card["evidence"]["threshold"] == 1.0
    assert card["evidence"]["feature"] == "MG_PASSTHROUGH_7D"


def test_below_direction_matches_low_retention():
    assert match_patterns({"MG_RETENTION_RATIO": 0.01})[0]["id"] == "low_retention"
    assert match_patterns({"MG_RETENTION_RATIO": 0.5}) == []


def test_confidence_grows_with_distance_past_the_threshold():
    near = match_patterns({"MG_PASSTHROUGH_7D": 1.05})[0]["confidence"]
    far = match_patterns({"MG_PASSTHROUGH_7D": 2.5})[0]["confidence"]
    assert 0.0 < near < far <= 1.0


def test_confidence_is_capped_at_one():
    card = match_patterns({"MG_BALANCE_DRAIN": 500.0})[0]
    assert card["confidence"] == 1.0


def test_missing_feature_is_not_a_match():
    """An absent measurement is not evidence of a pattern."""
    assert match_patterns({"MG_PASSTHROUGH_31D": 2.0,
                           "MG_PASSTHROUGH_7D": None}) == [
        c for c in match_patterns({"MG_PASSTHROUGH_31D": 2.0})]


def test_nan_and_non_numeric_values_are_skipped():
    assert match_patterns({"MG_PASSTHROUGH_7D": float("nan")}) == []
    assert match_patterns({"MG_PASSTHROUGH_7D": "high"}) == []


def test_cards_are_ordered_by_confidence():
    cards = match_patterns({"MG_PASSTHROUGH_7D": 1.05,
                            "MG_CASHOUT_PRESSURE": 1.0})
    confs = [c["confidence"] for c in cards]
    assert confs == sorted(confs, reverse=True)


def test_every_card_carries_the_no_score_contract():
    for card in match_patterns({"MG_PASSTHROUGH_7D": 2.0,
                                "MG_ALERT_CONVERGENCE": 6.0}):
        assert "did not contribute to the risk score" in card["contract"]


def test_only_direct_grade_typologies_are_defined():
    """PROXY rows (11-15 of the matrix) must not appear as detected patterns."""
    rows = {d.typology_row for d in PATTERN_DEFINITIONS}
    assert rows <= set(range(1, 11))


def test_every_definition_names_a_meta_feature():
    from muleguard.features.meta_features import META_DESCRIPTIONS

    for d in PATTERN_DEFINITIONS:
        assert d.feature in META_DESCRIPTIONS, f"{d.id} names an unknown feature"


def test_definitions_have_unique_ids():
    ids = [d.id for d in PATTERN_DEFINITIONS]
    assert len(ids) == len(set(ids))
