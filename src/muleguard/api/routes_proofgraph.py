"""Dual-Evidence ProofGraph endpoints.

    GET /v1/proofgraph/{case_id}   full graph: evidence for, against, doubt
    GET /v1/courtroom/{case_id}    just the prosecution/defence/verdict panel
    GET /v1/proofgraph/{case_id}/twin   nearest legitimate account, if indexed

Every node returned here names the artifact it came from. Nothing is generated
by a language model: the LLM narrator reads this graph, it never writes to it.
The endpoints run the traceability and language assertions before returning, so
an untraceable or unsafely-worded node fails the request rather than reaching a
reviewer's screen.
"""
from __future__ import annotations

import json

import numpy as np
from fastapi import APIRouter, HTTPException

from muleguard.api import database as db
from muleguard.explain import proofgraph as pg
from muleguard.explain.pattern_cards import match_patterns
from muleguard.logging import get_logger
from muleguard.models.scoring import load_bundle

log = get_logger("api.proofgraph")

router = APIRouter(tags=["proofgraph"])

_TWIN: pg.TwinIndex | None = None
_TWIN_TRIED = False
_REGISTRY: dict | None = None
_REGISTRY_TRIED = False


def _registry() -> dict | None:
    """The persisted feature dictionary, loaded once.

    Without it every evidence node renders as a bare ``F1813`` and the twin
    difference table shows an em dash where the variable name belongs. The
    dictionary is built from Description.xlsx and is the only sanctioned source
    of those names, so it is loaded here rather than reinvented in the route.
    Failure is not fatal: a graph labelled with column codes is still traceable
    evidence, and refusing the case would be the worse answer.
    """
    global _REGISTRY, _REGISTRY_TRIED
    if _REGISTRY is not None or _REGISTRY_TRIED:
        return _REGISTRY
    _REGISTRY_TRIED = True
    try:
        from muleguard.features.dictionary import load_registry

        _REGISTRY = load_registry()
    except Exception as exc:  # noqa: BLE001 - labels degrade, evidence does not
        log.warning("feature registry unavailable, nodes will show column codes: %s", exc)
        _REGISTRY = None
    return _REGISTRY


def _twin_index() -> pg.TwinIndex | None:
    """Build the legitimate-account index once, lazily.

    Lazily because it needs the development matrix, which costs a few seconds
    to assemble; a demo that never opens a case should not pay for it. Failure
    is not fatal - the Counterfactual Twin is one panel of the graph, and a
    graph without it is still complete evidence.
    """
    global _TWIN, _TWIN_TRIED
    if _TWIN is not None or _TWIN_TRIED:
        return _TWIN
    _TWIN_TRIED = True
    try:
        from muleguard.features.frame import build_model_frame
        from muleguard.models.harness import dev_split

        b = load_bundle()
        mf = build_model_frame(feature_subset=b["feature_list_kept"])
        dev = dev_split()
        _TWIN = pg.TwinIndex.from_matrix(
            mf.X[dev.row_index], mf.y[dev.row_index], list(mf.feature_names))
        log.info("twin index built over %d legitimate rows", _TWIN.X.shape[0])
    except Exception as exc:  # noqa: BLE001 - one optional panel, never fatal
        log.warning("twin index unavailable: %s", exc)
        _TWIN = None
    return _TWIN


def _load_case(case_id: str) -> tuple[dict, dict]:
    with db.connect() as c:
        case = c.execute("SELECT * FROM cases WHERE case_id=?",
                         (case_id,)).fetchone()
        if not case:
            raise HTTPException(404, "case not found")
        row = c.execute("SELECT payload_json FROM scores WHERE id=?",
                        (case["score_id"],)).fetchone()
    if not row:
        raise HTTPException(409, "case has no stored score payload")
    return dict(case), json.loads(row["payload_json"])


def _build(case_id: str, with_twin: bool) -> dict:
    case, score = _load_case(case_id)
    reasons = score.get("top_reasons") or []
    if not reasons:
        raise HTTPException(
            409, "this score was stored without explanations, so no evidence "
                 "graph can be built for it; rescore with explanations enabled")

    registry = _registry()

    twin_payload = None
    if with_twin:
        idx = _twin_index()
        if idx is not None:
            x_row, focus = _row_and_focus(score, reasons, idx)
            if x_row is not None:
                hit = idx.nearest(x_row, focus)
                if hit is not None:
                    j, dist = hit
                    twin_payload = {
                        "reference": idx.row_labels[j],
                        "distance": float(dist),
                        "differences": pg.twin_differences(
                            x_row, idx.X[j], idx.feature_names, focus,
                            registry=registry),
                    }

    # Semantic pattern cards. Matched from the account's own meta-feature
    # values against the DIRECT-grade typologies in the availability matrix -
    # they annotate the graph and never contributed to the score. Reason rows
    # are the only place the stored payload keeps values, so that is what they
    # are matched against.
    row_values = {r["feature"]: r.get("value") for r in reasons
                  if r.get("value") is not None}
    patterns = match_patterns(row_values)

    graph = pg.build_proofgraph(
        account_reference=case["account_reference"],
        score=score,
        reasons=reasons,
        registry=registry,
        twin=twin_payload,
        patterns=patterns,
        counterfactuals=score.get("counterfactual_twin") or None,
    )
    pg.assert_evidence_traceable(graph)
    pg.assert_language_safe(graph)
    graph["case_id"] = case_id
    return graph


def _row_and_focus(score: dict, reasons: list[dict],
                   idx: pg.TwinIndex) -> tuple[np.ndarray | None, list[int]]:
    """Recover the scored row vector and the columns the reasons point at.

    The stored payload keeps reason values but not the whole 3,900-wide row, so
    the twin comparison is done on the reason features only. That is the honest
    scope: those are the features the explanation is actually about.
    """
    pos = {n: i for i, n in enumerate(idx.feature_names)}
    focus, x = [], np.full(len(idx.feature_names), np.nan)
    for r in reasons:
        j = pos.get(r.get("feature"))
        if j is None:
            continue
        v = r.get("value")
        if v is None:
            continue
        x[j] = float(v)
        focus.append(j)
    if not focus:
        return None, []
    # Unnamed columns are filled with the legitimate median so the distance is
    # driven by the reason features rather than by absent data.
    med = np.nanmedian(idx.X, axis=0)
    x = np.where(np.isfinite(x), x, np.where(np.isfinite(med), med, 0.0))
    return x, focus


@router.get("/v1/proofgraph/{case_id}")
def proofgraph(case_id: str, twin: bool = True) -> dict:
    return _build(case_id, with_twin=twin)


@router.get("/v1/courtroom/{case_id}")
def courtroom(case_id: str) -> dict:
    g = _build(case_id, with_twin=False)
    return {
        "case_id": case_id,
        "account_reference": g["account_reference"],
        "calibrated_risk": g["calibrated_risk"],
        "risk_tier": g["risk_tier"],
        "disagreement": g["disagreement"],
        "courtroom": g["courtroom"],
        "evidence_counts": g["evidence_counts"],
        "note": ("Prosecution and defence are drawn from the same verified "
                 "evidence graph. No language model contributed a fact here."),
    }


@router.get("/v1/proofgraph/{case_id}/twin")
def twin_only(case_id: str) -> dict:
    g = _build(case_id, with_twin=True)
    twin_nodes = [n for n in g["nodes"] if n["type"] == pg.NODE_TWIN]
    if not twin_nodes:
        raise HTTPException(
            404, "no counterfactual twin is available for this case")
    return {"case_id": case_id, "twin": twin_nodes[0],
            "differences": twin_nodes[0].get("extra", {}).get("differences", [])}
