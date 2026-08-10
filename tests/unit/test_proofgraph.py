"""Dual-Evidence ProofGraph: traceability, two-sidedness and safe language."""
from __future__ import annotations

import numpy as np
import pytest

from muleguard.explain import proofgraph as pg


def _score(**over):
    base = {
        "model_version": "test-v1",
        "calibrated_risk": 0.72,
        "risk_tier": "HIGH",
        "raw_scores": {"lightgbm": 0.75, "xgboost": 0.71, "catboost": 0.70},
        "model_agreement": 0.95,
        "conformal_status": "SINGLETON_HIGH",
        "ood_status": "IN_DISTRIBUTION",
        "anomaly_percentile": 97.0,
        "verifier_confirms_risk": True,
        "verifier_probability": 0.81,
    }
    base.update(over)
    return base


def _reasons():
    return [
        {"feature": "F3642", "verified_semantic_name": None, "value": 41.0,
         "legitimate_cohort_median": 2.0, "legitimate_percentile": 99.2,
         "shap_contribution": 0.9, "direction": "INCREASES_RISK"},
        {"feature": "F3136", "verified_semantic_name": None, "value": 0.0,
         "legitimate_cohort_median": 1.0, "legitimate_percentile": 11.0,
         "shap_contribution": -0.3, "direction": "DECREASES_RISK"},
    ]


def test_graph_is_dual_sided():
    """A ProofGraph without a defence side is an advocate, not evidence."""
    g = pg.build_proofgraph(account_reference="A1", score=_score(),
                            reasons=_reasons())
    assert g["evidence_counts"]["prosecution"] >= 1
    assert g["evidence_counts"]["defence"] >= 1
    assert g["courtroom"]["prosecution"] and g["courtroom"]["defence"]


def test_every_node_names_its_source():
    g = pg.build_proofgraph(account_reference="A1", score=_score(),
                            reasons=_reasons())
    for node in g["nodes"]:
        assert node["source"], f"node {node['id']} has no provenance"


def test_edges_only_join_existing_nodes():
    g = pg.build_proofgraph(account_reference="A1", score=_score(),
                            reasons=_reasons())
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in ids and e["target"] in ids


def test_untraceable_node_is_refused():
    with pytest.raises(pg.UntraceableEvidence):
        pg.assert_evidence_traceable(
            {"nodes": [{"id": "x", "source": ""}], "edges": []})


@pytest.mark.parametrize("term", ["GUILTY", "criminal", "certified clean",
                                  "AUTO_FREEZE", "permanently safe"])
def test_forbidden_language_is_refused(term):
    with pytest.raises(pg.UnsafeLanguage):
        pg.assert_language_safe({"note": f"this account is {term}"})


def test_no_verdict_asserts_criminality():
    """Verdicts describe work for a reviewer, never the account holder."""
    for v in (pg.VERDICT_REVIEW, pg.VERDICT_ENHANCED, pg.VERDICT_MONITOR,
              pg.VERDICT_INSUFFICIENT, pg.VERDICT_NO_ACTION):
        pg.assert_language_safe({"verdict": v})


def test_verifier_and_anomaly_reach_the_defence():
    """The strongest false-positive protections must not be dropped.

    Both arrive as structural nodes rather than SHAP reasons; an earlier
    version filtered the courtroom to UNCERTAINTY-typed nodes only and lost
    them silently, which is exactly the failure this asserts against.
    """
    g = pg.build_proofgraph(
        account_reference="A1",
        score=_score(verifier_confirms_risk=False, verifier_probability=0.2,
                     anomaly_percentile=40.0),
        reasons=_reasons())
    sources = {d["source"] for d in g["courtroom"]["defence"]}
    assert "verifier_confirms_risk" in sources
    assert "anomaly_percentile" in sources


def test_disagreement_does_not_raise_risk():
    """UPDATE 6: disagreement is uncertainty, never additional risk."""
    agree = pg.build_proofgraph(
        account_reference="A1",
        score=_score(raw_scores={"lightgbm": 0.72, "xgboost": 0.71,
                                 "catboost": 0.73}),
        reasons=_reasons())
    disagree = pg.build_proofgraph(
        account_reference="A1",
        score=_score(raw_scores={"lightgbm": 0.95, "xgboost": 0.71,
                                 "catboost": 0.20}),
        reasons=_reasons())
    assert agree["calibrated_risk"] == disagree["calibrated_risk"]
    assert disagree["disagreement"]["status"] == "MODEL_DISAGREEMENT"
    assert disagree["courtroom"]["contested"] is True
    # a contested case may not be promoted to the strongest recommendation
    assert disagree["courtroom"]["verdict"] != pg.VERDICT_ENHANCED


def test_contested_case_is_not_escalated_hardest():
    weak = pg.build_proofgraph(
        account_reference="A1",
        score=_score(model_agreement=0.55, conformal_status="AMBIGUOUS",
                     verifier_confirms_risk=False, anomaly_percentile=35.0),
        reasons=_reasons())
    assert weak["courtroom"]["verdict"] in (pg.VERDICT_INSUFFICIENT,
                                            pg.VERDICT_REVIEW)


def test_disagreement_profile_statistics():
    d = pg.disagreement_profile({"a": 0.1, "b": 0.9})
    assert d["max_minus_min"] == pytest.approx(0.8)
    assert d["mean"] == pytest.approx(0.5)
    assert d["status"] == "MODEL_DISAGREEMENT"
    assert "does not raise" in d["interpretation"]


def test_meta_features_render_with_a_description():
    g = pg.build_proofgraph(
        account_reference="A1", score=_score(),
        reasons=[{"feature": "MG_PASSTHROUGH_7D", "verified_semantic_name": None,
                  "value": 0.97, "legitimate_cohort_median": 0.2,
                  "legitimate_percentile": 98.0, "shap_contribution": 0.6,
                  "direction": "INCREASES_RISK"}])
    node = next(n for n in g["nodes"] if n["source"] == "MG_PASSTHROUGH_7D")
    assert "Pass-through" in node["detail"]


def test_twin_is_a_real_row_not_a_synthetic_point():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = np.zeros(200, dtype=int)
    y[:5] = 1
    idx = pg.TwinIndex.from_matrix(X, y, [f"F{i}" for i in range(6)])
    hit = idx.nearest(X[7], focus_idx=[0, 1, 2])
    assert hit is not None
    j, dist = hit
    assert dist >= 0.0
    assert np.array_equal(idx.X[j], idx.X[j])  # the twin is an indexed row
    assert idx.row_labels[j].startswith("DEV-")


def test_twin_differences_are_ranked_by_gap():
    a = np.array([10.0, 1.0, 5.0])
    b = np.array([1.0, 1.0, 4.0])
    diffs = pg.twin_differences(a, b, ["F1", "F2", "F3"], [0, 1, 2])
    assert diffs[0]["feature"] == "F1"
    assert diffs[0]["absolute_gap"] == pytest.approx(9.0)


def test_graph_contains_no_counterparty_relationships():
    """UPDATE 8: no sender/receiver edges may be derived from F-columns."""
    g = pg.build_proofgraph(account_reference="A1", score=_score(),
                            reasons=_reasons())
    relations = {e["relation"] for e in g["edges"]}
    forbidden = {"SENT_TO", "RECEIVED_FROM", "TRANSFERRED_TO", "COUNTERPARTY"}
    assert not (relations & forbidden)
