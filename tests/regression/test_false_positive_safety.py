"""Section 31: the case where a retrieval layer does real damage.

A high-volume legitimate merchant looks, in aggregate features, a great deal
like a mule account: many counterparties, high throughput, unusual velocity.
That is precisely why the Cohort Radar is dangerous around merchants - a panel
headed "behaviourally similar accounts" next to a merchant's name is an
accusation if anything downstream treats it as evidence.

Section 31 sets five requirements: show the similarity, show the exculpatory
evidence, do not raise the classifier risk, do not auto-escalate, keep human
review semantics. This file checks all five, and covers section 37 cases 12, 13
and 14 - the merchant case, and the two directions in which similarity and
model risk disagree.
"""
from __future__ import annotations

import copy

import numpy as np
import polars as pl
import pytest

from muleguard import settings
from muleguard.usp import cohort_radar as cr

BUNDLE = settings.MODELS_DIR / "final_bundle.joblib"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="final bundle not built")

#: The bands, taken from the module rather than retyped. A test that spells a
#: band name out is a test that passes when the product renames one.
STRONG_BANDS = (cr.BAND_VERY_HIGH, cr.BAND_HIGH)
ALL_BANDS = STRONG_BANDS + (cr.BAND_MODERATE, cr.BAND_TYPICAL)


def _pos(index, row: int) -> int:
    """Position of a reference row in the index arrays.

    ``CohortIndex.position_of`` resolves an *account reference string*; the
    arrays are keyed by sorted row id, so a row lookup is a search, not an
    index.
    """
    pos = int(np.searchsorted(index.row_index, row))
    assert index.row_index[pos] == row, f"row {row} is not in the reference frame"
    return pos


needs_index = pytest.mark.skipif(
    not cr.TRANSFORM_PATH.exists(),
    reason="cohort transform not built - python -m muleguard.cli.build_cohort_radar")


@pytest.fixture(scope="module")
def index():
    return cr.build_index()


@pytest.fixture(scope="module")
def scored_reference(index):
    """The 600 riskiest reference accounts, scored through the live path.

    The riskiest end is where section 31 lives: a merchant only becomes a
    false-positive problem once the classifier has already put it in a review
    queue. Scoring the quiet tail would test nothing.
    """
    from muleguard.features.frame import raw_with_meta
    from muleguard.models.scoring import score_rows

    order = np.argsort(-index.risk)[:600]
    rows = index.row_index[order].tolist()
    scored = score_rows(raw_with_meta()[rows], with_explanations=False,
                        with_counterfactual=False)
    return list(zip(rows, scored))


@pytest.fixture(scope="module")
def merchant_like(scored_reference):
    """A high-volume account the verifier reads as a genuine business.

    Selected by the pipeline's own merchant verifier - ``STRONG_BUSINESS_
    EVIDENCE`` is its top band - rather than by a hand-picked row id. Picking a
    row by eye would test that row; picking by the verifier's verdict tests the
    condition section 31 is actually about.
    """
    strong = [(r, s) for r, s in scored_reference
              if s["merchant_safeguard"]["merchant_band"] == "STRONG_BUSINESS_EVIDENCE"]
    if not strong:
        pytest.skip("no strong-business-evidence account among the riskiest rows")
    strong.sort(key=lambda rs: -float(rs[1]["calibrated_risk"]))
    return strong[0]


@pytest.fixture(scope="module")
def merchant_cohort(index, merchant_like):
    row, _ = merchant_like
    return cr.cohort_for_row(row, k=10)


# --------------------------------------------------------------------------
# section 31: the five requirements
# --------------------------------------------------------------------------


@needs_index
def test_the_similarity_is_displayed_not_suppressed(merchant_cohort):
    """Hiding the panel for merchants would be the wrong fix.

    An analyst who cannot see that a merchant resembles a flagged account
    cannot rule the resemblance out either. The requirement is disclosure with
    context, not silence.
    """
    neighbors = merchant_cohort["neighbors"]
    assert neighbors, "the merchant-like account was given no cohort at all"
    for n in neighbors:
        assert 0.0 <= n["behavioral_similarity"] <= 1.0
        assert n["similarity_band"] in ALL_BANDS
        assert n["main_shared_features"], "similarity shown without saying why"


@needs_index
def test_the_exculpatory_evidence_travels_with_the_account(merchant_like):
    """The business evidence must reach the analyst, and must not clear anyone."""
    _, score = merchant_like
    safeguard = score["merchant_safeguard"]
    assert safeguard["merchant_band"] == "STRONG_BUSINESS_EVIDENCE"
    assert safeguard["guarantees"], "the safeguard shipped without its guarantees"
    # Mode A: exculpatory evidence adjusts escalation confidence, never the score.
    assert safeguard["before_score"] == pytest.approx(
        safeguard["after_policy_score"], abs=1e-12)
    assert safeguard["delta"] == 0.0


@needs_index
def test_the_exculpatory_evidence_appears_in_the_proofgraph(merchant_like):
    """Section 31: shown as defence, next to the similarity, not buried."""
    from muleguard.explain.proofgraph import (
        NODE_EVIDENCE_AGAINST, NODE_UNCERTAINTY, build_proofgraph)

    row, score = merchant_like
    graph = build_proofgraph(account_reference=f"RV-{row:06d}", score=score,
                             reasons=[])
    defence = [n for n in graph["nodes"]
               if n["type"] in (NODE_EVIDENCE_AGAINST, NODE_UNCERTAINTY)]
    assert defence, "a strong-business-evidence account with no defence node"
    assert graph["courtroom"]["defence"], "the courtroom heard no defence"
    assert graph["control_attribution"]["automatic_actions_permitted"] == []


@needs_index
def test_the_cohort_lookup_does_not_move_the_classifier_risk(index, merchant_like):
    """The load-bearing one. Score, look up the cohort, score again.

    Bit-identity rather than a tolerance: the cohort code is not in the scoring
    path at all, so anything other than an exact match means it has become so.
    """
    from muleguard.features.frame import raw_with_meta
    from muleguard.models.scoring import score_rows

    row, before = merchant_like
    frame = raw_with_meta()
    cohort = cr.cohort_for_row(row, k=25)
    assert cohort["neighbors"]
    after = score_rows(frame[[row]], with_explanations=False,
                       with_counterfactual=False)[0]
    assert after["calibrated_risk"] == before["calibrated_risk"]
    assert after["risk_tier"] == before["risk_tier"]
    assert after["raw_scores"] == before["raw_scores"]
    # And the cohort payload reports the classifier's figure, not its own.
    assert cohort["risk_probability"] == pytest.approx(
        float(index.risk[_pos(index, row)]), abs=1e-12)


@needs_index
def test_no_neighbour_is_escalated_by_appearing_in_the_cohort(index, merchant_cohort):
    """Section 23/31: membership must not change anyone else's tier either."""
    for n in merchant_cohort["neighbors"]:
        pos = _pos(index, n["row_index"])
        assert n["neighbor_risk_probability"] == pytest.approx(
            float(index.risk[pos]), abs=1e-12)
        assert n["neighbor_risk_tier"] == index.tier[pos]


@needs_index
def test_the_panel_states_that_it_changes_nothing(merchant_cohort):
    """Human review semantics, in the words the analyst actually reads."""
    assert merchant_cohort["action_policy"] == cr.ACTION_POLICY
    assert "does not change" in merchant_cohort["action_policy"]
    assert merchant_cohort["disclaimer"] == cr.DISCLAIMER
    assert merchant_cohort.get("affects_model_score") in (False, None)
    cr.assert_language_safe(merchant_cohort)


@needs_index
def test_case_12_a_merchant_like_case_carries_no_automatic_action(merchant_like):
    """Section 37 case 12. Nothing in the payload authorises an action."""
    from muleguard.usp.control_attribution import NEVER_AUTOMATIC, control_attribution

    _, score = merchant_like
    card = control_attribution(risk_probability=score["calibrated_risk"],
                              risk_tier=score["risk_tier"])
    assert card["automatic_actions_permitted"] == []
    assert card["affects_model_output"] is False
    blob = repr(card).upper()
    for action in NEVER_AUTOMATIC:
        assert f'"{action}"' not in blob


# --------------------------------------------------------------------------
# section 37 cases 13 and 14: when similarity and risk disagree
# --------------------------------------------------------------------------


def _query_with(index, *, want_high_similarity: bool, want_high_risk: bool,
                probe: int = 400):
    """Find a reference row whose cohort exhibits the requested disagreement.

    Searched rather than hard-coded. A row id written into the test would stop
    describing the condition the moment the transform is refitted, and the test
    would keep passing while measuring something else.
    """
    order = (np.argsort(-index.risk) if want_high_risk
             else np.argsort(index.risk))[:probe]
    for pos in order.tolist():
        row = int(index.row_index[pos])
        result = cr.cohort_for_row(row, k=5, with_explanations=False)
        top = result["neighbors"]
        if not top:
            continue
        strong = top[0]["similarity_band"] in STRONG_BANDS
        if strong is want_high_similarity:
            return row, result
    pytest.skip(f"no row found with high_similarity={want_high_similarity}, "
                f"high_risk={want_high_risk} in the first {probe} probed")


@needs_index
def test_case_13_high_similarity_to_a_risky_account_leaves_a_quiet_one_quiet(index):
    """Section 37 case 13. This is the accusation-by-association failure mode.

    A low-risk account that closely resembles a flagged one must keep its low
    risk and its low tier. Similarity is retrieval; only the classifier assigns
    risk, and it was never asked.
    """
    row, result = _query_with(index, want_high_similarity=True, want_high_risk=False)
    pos = _pos(index, row)
    assert result["risk_probability"] == pytest.approx(float(index.risk[pos]), abs=1e-12)
    assert result["risk_tier"] == index.tier[pos]
    top = result["neighbors"][0]
    assert top["similarity_band"] in STRONG_BANDS
    # Re-querying after the neighbours are known must not move anything.
    again = cr.cohort_for_row(row, k=5, with_explanations=False)
    assert again["risk_probability"] == result["risk_probability"]
    assert again["risk_tier"] == result["risk_tier"]
    # The summary may say the neighbourhood is risky. The account's own figure
    # must be untouched by that.
    assert result["cohort_summary"]["max_neighbor_risk"] >= 0.0
    assert result["risk_probability"] == pytest.approx(float(index.risk[pos]), abs=1e-12)


@needs_index
def test_case_14_a_risky_account_with_no_close_neighbours_stays_risky(index):
    """Section 37 case 14. The mirror image, and the one that gets forgotten.

    A radar that quietly discounts an alert because nothing resembles it would
    be suppressing true positives - the expensive direction of this error.
    """
    row, result = _query_with(index, want_high_similarity=False, want_high_risk=True)
    pos = _pos(index, row)
    assert result["risk_probability"] == pytest.approx(float(index.risk[pos]), abs=1e-12)
    assert result["risk_tier"] == index.tier[pos]
    assert result["neighbors"][0]["similarity_band"] in (
        cr.BAND_MODERATE, cr.BAND_TYPICAL)
    assert result["risk_tier"] != "MONITOR", "the probe did not find a risky row"


@needs_index
def test_the_cohort_summary_never_reports_a_risk_of_its_own(index, merchant_cohort):
    """Every number in the summary must be a statistic *of the neighbours*.

    The forbidden thing is a blended figure - `0.8*model + 0.2*cohort` and its
    relatives. The way to prove one is absent is to show every reported value
    is reproducible from the neighbour list alone.
    """
    summary = merchant_cohort["cohort_summary"]
    risks = [n["neighbor_risk_probability"] for n in merchant_cohort["neighbors"]]
    assert summary["n_neighbors"] == len(risks)
    assert summary["max_neighbor_risk"] == pytest.approx(max(risks), abs=1e-9)
    assert summary["median_neighbor_risk"] == pytest.approx(
        float(np.median(risks)), abs=1e-9)
    assert "combined" not in summary and "final_risk" not in summary
    assert "adjusted_risk" not in merchant_cohort
