"""MuleGuard FastAPI application.

Endpoints (all JSON, OpenAPI at /docs):
  GET  /health/live, /health/ready
  GET  /v1/model
  POST /v1/score            single account (raw features dict)
  POST /v1/score/batch      up to `batch_limit` accounts
  GET  /v1/cases            queue with filters
  GET  /v1/cases/{id}
  POST /v1/cases/{id}/decision     analyst action (identity + reason required)
  POST /v1/cases/{id}/feedback     verdict feedback loop
  GET  /v1/metrics/summary  numbers straight from artifacts (never invented)
  GET  /v1/capacity/curve   measured recall/precision per analyst review budget
  POST /v1/capacity/plan    one capacity question -> advisory threshold
  GET  /v1/drift/status
  POST /v1/reports/{case_id}/generate   evidence packet (+optional LLM prose)

Design rules enforced here:
  - masked account references only; no PII
  - the LLM never touches a score; narrative generation is separate and optional
  - a missing selected feature -> 422 SCHEMA_ERROR (no silent fill)
  - scoring works identically with Ollama absent
  - rate limiting sized for a demo; correlation IDs on every request
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from muleguard import settings
from muleguard.api import database as db
from muleguard.llm.ollama_client import OllamaNarrator
from muleguard.llm.schemas import NarratorInput, ReasonFact
from muleguard.logging import get_logger
from muleguard.models.scoring import SchemaError, load_bundle, score_rows
from muleguard.utils import load_json

log = get_logger("api.main")

app = FastAPI(
    title="MuleGuard - Trinetra",
    version="1.0.0",
    description="AI/ML classification of suspicious mule accounts. "
                "Behavioural risk scoring with human-in-the-loop review. "
                "Scores are not proof of criminal intent.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

from muleguard.api.routes_capacity import router as capacity_router  # noqa: E402
from muleguard.api.routes_graph import router as graph_router  # noqa: E402
from muleguard.api.routes_proofgraph import router as proofgraph_router  # noqa: E402
from muleguard.api.routes_upload import router as upload_router  # noqa: E402
from muleguard.api.routes_validation import router as validation_router  # noqa: E402

app.include_router(upload_router)
app.include_router(proofgraph_router)
app.include_router(validation_router)
app.include_router(graph_router)
app.include_router(capacity_router)

_narrator: OllamaNarrator | None = None
_rate: dict[str, list[float]] = {}
RATE_LIMIT_PER_MIN = 240
BATCH_LIMIT = 500
MASKED_REF = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def get_narrator() -> OllamaNarrator:
    global _narrator
    if _narrator is None:
        _narrator = OllamaNarrator()
    return _narrator


def _verify_bundle_hash(manifest: dict) -> str:
    """Re-hash the bundle on disk and compare it with the manifest.

    Recording the manifest's hash in the database without ever recomputing it
    proves only that the manifest is self-consistent. The question that matters
    at boot is whether the file being served is still the file that was
    evaluated, and the only way to answer it is to read the bytes.
    """
    import hashlib

    path = settings.MODELS_DIR / "final_bundle.joblib"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    expected = str(manifest.get("bundle_sha256", ""))
    if expected and actual != expected:
        log.error("MODEL INTEGRITY: %s hashes to %s but the manifest records %s",
                  path.name, actual[:16], expected[:16])
    else:
        log.info("model integrity verified: %s sha256=%s", path.name, actual[:16])
    return actual


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    try:
        b = load_bundle()  # warmup
        manifest = load_json(settings.MODELS_DIR / "model_manifest.json")
        _verify_bundle_hash(manifest)
        with db.connect() as c:
            row = c.execute(
                "SELECT 1 FROM model_versions WHERE bundle_sha256=?",
                (manifest["bundle_sha256"],),
            ).fetchone()
            if not row:
                c.execute(
                    "INSERT INTO model_versions (version, bundle_sha256, winner, created_utc)"
                    " VALUES (?,?,?,?)",
                    (b["version"], manifest["bundle_sha256"], b["winner_oof_name"], db.utcnow()),
                )
        log.info("startup complete, model warm")
    except FileNotFoundError:
        log.warning("no final bundle yet - scoring endpoints will 503")


def rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "anon"
    now = time.monotonic()
    window = [t for t in _rate.get(key, []) if now - t < 60]
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "rate limit exceeded")
    window.append(now)
    _rate[key] = window


# ---------- schemas ----------
class ScoreRequest(BaseModel):
    account_reference: str = Field(min_length=1, max_length=64,
                                   description="masked/synthetic id only")
    features: dict[str, Any] = Field(description="raw feature name -> value")
    schema_version: str = "1.0"


class BatchScoreRequest(BaseModel):
    accounts: list[ScoreRequest] = Field(max_length=BATCH_LIMIT)


class DecisionRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=64)
    action: str = Field(pattern="^(ASSIGN|MARK_REVIEWED|RECOMMEND_FREEZE|ESCALATE|CLOSE|REOPEN)$")
    reason: str = Field(min_length=5, max_length=2000)
    approved_by: str | None = None


class FeedbackRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=64)
    verdict: str = Field(pattern="^(CONFIRMED_MULE|FALSE_POSITIVE|INCONCLUSIVE)$")
    notes: str | None = Field(default=None, max_length=4000)


# ---------- helpers ----------
def _prefer_v2(path):
    """The v2 sibling of an artifact when one exists, else the original.

    ``lens_stack_oof.json`` becomes ``lens_stack_oof_v2.json``. The v2 files are
    the ones the frozen bundle came out of; the originals belong to an earlier
    tournament and describe a model that is not deployed.
    """
    v2 = path.with_name(f"{path.stem}_v2{path.suffix}")
    return v2 if v2.exists() else path


def _annotate_locked_test(block: dict) -> dict:
    """Say, in the payload itself, which scorer produced the locked-test figures.

    ``locked_test_metrics.json`` is the July run of ``catboost_tuned_top60``,
    which the Feature Availability Firewall later retired; the deployed bundle
    is ``xgboost_top_120`` and scores lower on the same split. The repository
    already knows this - ``holdout_metrics.json`` separates the champion from
    the retired run and warns against reading the latter as current behaviour -
    but the dashboard was reading the retired file with no such warning, so its
    headline PR-AUC belonged to a model that is not served. The artifact is
    still returned unchanged (its tier distribution and calibration blocks have
    no v2 equivalent); what is added is the provenance needed to stop a reader
    attributing the number to the deployed model.
    """
    path = settings.METRICS_DIR / "holdout_metrics.json"
    if not path.exists():
        return block
    holdout = load_json(path) or {}
    champion = holdout.get("current_champion") or {}
    if not champion:
        return block
    same = str(champion.get("model") or "") in str(block.get("split") or "")
    block["describes_deployed_model"] = bool(same)
    if not same:
        block["retired_warning"] = (holdout.get("retired_run") or {}).get("warning")
        block["deployed_scorer_result"] = {
            k: champion.get(k) for k in
            ("model", "split", "source", "n", "n_positives", "prevalence",
             "pr_auc", "roc_auc", "lift_over_prevalence", "mcc_at_top_100",
             "at_budget")
        }
    return block


def _validate_ref(ref: str) -> str:
    if not MASKED_REF.match(ref):
        raise HTTPException(422, "account_reference must be a masked id (alnum/_/-)")
    return ref


def _score_one(req: ScoreRequest, correlation_id: str) -> dict[str, Any]:
    _validate_ref(req.account_reference)
    if settings.TARGET_COLUMN in req.features:
        raise HTTPException(422, f"target column {settings.TARGET_COLUMN} must not "
                                 "be present in a scoring request")
    try:
        rows = pl.DataFrame([req.features], infer_schema_length=None)
        results = score_rows(rows)
    except SchemaError as e:
        raise HTTPException(422, f"SCHEMA_ERROR: {e}")
    result = results[0]
    request_id, score_id, case_id = db.record_score(
        req.account_reference, result, correlation_id
    )
    return {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "account_reference": req.account_reference,
        "case_id": case_id,
        **{k: v for k, v in result.items() if k != "ood_detail"},
    }


# ---------- endpoints ----------
@app.get("/health/live")
def health_live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> dict:
    try:
        b = load_bundle()
    except FileNotFoundError:
        raise HTTPException(503, "model bundle not available")
    narrator = get_narrator()
    return {
        "status": "ready",
        "model_version": b["version"],
        "winner": b["winner_oof_name"],
        "ollama_model": narrator.available_model(),
        "ollama_required": False,
    }


@app.get("/v1/model")
def model_info() -> dict:
    manifest_path = settings.MODELS_DIR / "model_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(503, "no model manifest")
    manifest = load_json(manifest_path)
    actual = _verify_bundle_hash(manifest)
    return {
        **manifest,
        "bundle_sha256_recomputed": actual,
        "integrity": ("VERIFIED" if actual == manifest.get("bundle_sha256")
                      else "MISMATCH"),
    }


@app.post("/v1/score")
def score(req: ScoreRequest, request: Request, _=Depends(rate_limit)) -> dict:
    correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    return _score_one(req, correlation_id)


@app.post("/v1/score/batch")
def score_batch(batch: BatchScoreRequest, request: Request, _=Depends(rate_limit)) -> dict:
    correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    results, errors = [], []
    for i, req in enumerate(batch.accounts):
        try:
            results.append(_score_one(req, correlation_id))
        except HTTPException as e:
            errors.append({"index": i, "account_reference": req.account_reference,
                           "error": e.detail})
    return {"correlation_id": correlation_id, "n_scored": len(results),
            "n_errors": len(errors), "results": results, "errors": errors}


@app.get("/v1/cases")
def list_cases(status: str | None = None, tier: str | None = None,
               limit: int = 100) -> dict:
    q = "SELECT * FROM cases"
    conds, params = [], []
    if status:
        conds.append("status = ?"); params.append(status)
    if tier:
        conds.append("risk_tier = ?"); params.append(tier)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY calibrated_risk DESC LIMIT ?"
    params.append(min(limit, 500))
    with db.connect() as c:
        rows = [dict(r) for r in c.execute(q, params).fetchall()]
    return {"cases": rows, "count": len(rows)}


@app.get("/v1/cases/{case_id}")
def case_detail(case_id: str) -> dict:
    with db.connect() as c:
        case = c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not case:
            raise HTTPException(404, "case not found")
        score = c.execute("SELECT payload_json FROM scores WHERE id=?",
                          (case["score_id"],)).fetchone()
        actions = [dict(r) for r in c.execute(
            "SELECT * FROM case_actions WHERE case_id=? ORDER BY id", (case_id,))]
        feedback = [dict(r) for r in c.execute(
            "SELECT * FROM analyst_feedback WHERE case_id=? ORDER BY id", (case_id,))]
    return {
        "case": dict(case),
        "score": json.loads(score["payload_json"]) if score else None,
        "actions": actions,
        "feedback": feedback,
    }


@app.post("/v1/cases/{case_id}/decision")
def case_decision(case_id: str, req: DecisionRequest) -> dict:
    if req.action == "RECOMMEND_FREEZE" and not req.approved_by:
        raise HTTPException(422, "RECOMMEND_FREEZE requires approved_by (authorised analyst). "
                                 "MuleGuard never freezes accounts automatically.")
    with db.connect() as c:
        case = c.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not case:
            raise HTTPException(404, "case not found")
        before = dict(case)
        c.execute(
            "INSERT INTO case_actions (case_id, actor, action, reason, requires_approval,"
            " approved_by, created_utc) VALUES (?,?,?,?,?,?,?)",
            (case_id, req.actor, req.action, req.reason,
             int(req.action == "RECOMMEND_FREEZE"), req.approved_by, db.utcnow()),
        )
        new_status = {"CLOSE": "CLOSED", "REOPEN": "OPEN",
                      "MARK_REVIEWED": "REVIEWED"}.get(req.action, case["status"])
        c.execute("UPDATE cases SET status=?, assignee=?, updated_utc=? WHERE case_id=?",
                  (new_status, req.actor if req.action == "ASSIGN" else case["assignee"],
                   db.utcnow(), case_id))
    db.audit("CASE_DECISION", req.actor, before=before,
             after={"case_id": case_id, "action": req.action, "status": new_status},
             detail={"reason": req.reason, "approved_by": req.approved_by})
    return {"case_id": case_id, "action": req.action, "status": new_status}


@app.post("/v1/cases/{case_id}/feedback")
def case_feedback(case_id: str, req: FeedbackRequest) -> dict:
    with db.connect() as c:
        if not c.execute("SELECT 1 FROM cases WHERE case_id=?", (case_id,)).fetchone():
            raise HTTPException(404, "case not found")
        c.execute(
            "INSERT INTO analyst_feedback (case_id, actor, verdict, notes, created_utc)"
            " VALUES (?,?,?,?,?)",
            (case_id, req.actor, req.verdict, req.notes, db.utcnow()),
        )
    db.audit("FEEDBACK", req.actor, after={"case_id": case_id, "verdict": req.verdict})
    return {"case_id": case_id, "verdict": req.verdict, "recorded": True}


@app.get("/v1/metrics/summary")
def metrics_summary() -> dict:
    """Numbers straight from the artifacts of the model that is deployed.

    Where a v2 artifact exists it wins, because v2 is the run that produced the
    frozen bundle. Serving the v1 files here quietly showed the dashboard a
    different model's numbers: a higher OOF PR-AUC than the deployed model
    earns, and a selection-frequency chart topped by quarantined columns that
    the shipped feature list does not contain. Both are worse than showing
    nothing.
    """
    out: dict[str, Any] = {}
    for name, path in [
        ("oof", settings.METRICS_DIR / "oof_metrics.json"),
        ("locked_test", settings.METRICS_DIR / "locked_test_metrics.json"),
        ("lens_stack_oof", _prefer_v2(settings.METRICS_DIR / "lens_stack_oof.json")),
        ("ensemble_decision", settings.METRICS_DIR / "ensemble_decision.json"),
        ("leakage_ablation", settings.METRICS_DIR / "with_vs_without_f3912.json"),
        ("registry", settings.REGISTRY_DIR / "registry.json"),
    ]:
        if path.exists():
            out[name] = load_json(path)
    if isinstance(out.get("locked_test"), dict):
        out["locked_test"] = _annotate_locked_test(out["locked_test"])
    freq_path = _prefer_v2(settings.FEATURES_DIR / "selection_frequency.csv")
    if freq_path.exists():
        freq = pl.read_csv(freq_path).head(60)
        out["selection_frequency"] = freq.to_dicts()
    if not out:
        raise HTTPException(503, "no metrics artifacts yet")
    return out


@app.get("/v1/drift/status")
def drift_status() -> dict:
    with db.connect() as c:
        row = c.execute(
            "SELECT snapshot_json, created_utc FROM drift_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row:
        return {"latest": json.loads(row["snapshot_json"]), "created_utc": row["created_utc"]}
    path = settings.METRICS_DIR / "drift_baseline.json"
    if path.exists():
        return {"latest": load_json(path), "created_utc": None}
    return {"latest": None, "note": "no drift snapshots recorded yet"}


@app.post("/v1/reports/{case_id}/generate")
def generate_report(case_id: str, use_llm: bool = True) -> dict:
    detail = case_detail(case_id)
    score = detail["score"]
    if score is None:
        raise HTTPException(409, "case has no stored score payload")
    ctx = NarratorInput(
        account_reference=detail["case"]["account_reference"],
        calibrated_risk=score["calibrated_risk"],
        risk_tier=score["risk_tier"],
        model_agreement=score["model_agreement"],
        conformal_status=score["conformal_status"],
        ood_status=score["ood_status"],
        top_reasons=[ReasonFact(**{k: v for k, v in r.items()
                                   if k in ReasonFact.model_fields})
                     for r in score.get("top_reasons", [])],
    )
    narrator = get_narrator()
    if use_llm:
        narrative = narrator.narrate(ctx)
    else:
        from muleguard.llm.deterministic_fallback import deterministic_narrative
        narrative = {"source": "deterministic",
                     "narrative": deterministic_narrative(ctx).model_dump(),
                     "llm_rejected_reasons": [], "model": None}
    packet = {
        "case_id": case_id,
        "generated_utc": db.utcnow(),
        "account_reference": detail["case"]["account_reference"],
        "decision_tier": score["risk_tier"],
        "risk_and_uncertainty": {
            "calibrated_risk": score["calibrated_risk"],
            "model_agreement": score["model_agreement"],
            "conformal_status": score["conformal_status"],
            "ood_status": score["ood_status"],
            "anomaly_percentile": score["anomaly_percentile"],
        },
        "verified_reasons": score.get("top_reasons", []),
        "narrative": narrative,
        "analyst_actions": detail["actions"],
        "model_version": score["model_version"],
        "limitations": score["limitations"],
    }
    with db.connect() as c:
        c.execute(
            "INSERT INTO generated_reports (case_id, kind, content_json, created_utc)"
            " VALUES (?,?,?,?)",
            (case_id, "evidence_packet", json.dumps(packet), db.utcnow()),
        )
    db.audit("REPORT_GENERATED", "system", after={"case_id": case_id,
                                                  "narrative_source": narrative["source"]})
    return packet
