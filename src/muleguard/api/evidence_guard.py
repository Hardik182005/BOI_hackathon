"""The one boundary between a stored score and evidence shown to a reviewer.

A stored score outlives the model that wrote it. That is the whole incident this
module exists for: on 2026-08-26 a ProofGraph presented ``F3898
MIN_RESOLVE_DAYS`` and ``F3914 FALSE_POSITIVE`` as prosecution evidence. Neither
column is in the production model. Both were in the *retired* CatBoost bundle
(model_version 1.0.0, frozen 2026-07-10, before the Feature Availability
Firewall existed), and 77 payloads it wrote are still in the database. A route
that reads ``scores.payload_json`` and renders it learns nothing about which
model produced it, so every such route was serving pre-firewall explanations as
though they were current findings.

The fix is not a filter. Stripping the two columns out of a retired payload
would leave a graph that *looks* current, built from a feature list that no
longer exists, with weights from a calibrator that is no longer in production -
a quieter version of the same lie. So the rule here is provenance first:

1. **Provenance.** Evidence is admissible as *current* only if the payload's
   ``model_version`` matches the loaded bundle's. Anything else is RETIRED and
   is never rendered as a finding, however clean its columns happen to be.
2. **Admissibility.** Independently, no quarantined column may leave any
   evidence surface, on either side of the argument. A current payload that
   somehow named one is a live bug, and is refused rather than shown.

Nothing is deleted. A retired payload stays exactly as written - it is the
record of what the system said on the day it said it, and it is the only trail
back to an incident like this one. It is served through
``/v1/proofgraph/{case}/provenance`` as a labelled audit record, in a shape no
reviewer can mistake for a case against an account.

Both gates live here rather than in each route because there were three routes
reading these payloads and only one of them checked anything.
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from muleguard.features import firewall
from muleguard.logging import get_logger
from muleguard.models.scoring import load_bundle

log = get_logger("api.evidence_guard")

PROVENANCE_CURRENT = "CURRENT"
PROVENANCE_RETIRED = "RETIRED"

#: What a refused payload is called everywhere it surfaces, so the UI, the API
#: and the release gate use one word for one condition.
RETIRED_ERROR = "RETIRED_EVIDENCE"
INADMISSIBLE_ERROR = "INADMISSIBLE_EVIDENCE"


def current_model_version() -> str:
    """The version of the bundle actually loaded, not a manifest's claim of it.

    A manifest is a file someone can edit; the bundle is what scores. If it
    cannot be loaded the honest answer is 503 - no score can be confirmed
    current - and never a guess that lets a stale payload through.
    """
    try:
        return str(load_bundle()["version"])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            503, "model bundle unavailable, so no score can be confirmed "
                 f"current: {exc}") from exc


def provenance(score: dict[str, Any]) -> dict[str, Any]:
    """Whether this stored payload came from the model now in production.

    Derived at read time from the payload's own ``model_version`` stamp rather
    than written into a column, for two reasons: it costs nothing to compute,
    and it re-derives itself correctly the next time the champion is replaced.
    A status column would have to be migrated on every promotion, and would be
    wrong - silently - in the window before someone remembered to.
    """
    stored = score.get("model_version")
    current = current_model_version()
    return {
        "status": PROVENANCE_CURRENT if stored == current else PROVENANCE_RETIRED,
        "stored_model_version": stored,
        "current_model_version": current,
    }


#: Keys known to carry per-feature model reasoning. Dropped by name because
#: they are the ones that exist today; the sweep below is what makes the
#: guarantee hold for the ones that get added tomorrow.
_EVIDENCE_KEYS = ("top_reasons", "counterfactual_twin")


def _quarantined_tokens() -> frozenset[str]:
    cfg = firewall.config()
    return frozenset(set(cfg.hard_quarantine) | set(cfg.conditional_quarantine)
                     | set(cfg.fairness_excluded))


def _names_a_quarantined_column(value: Any) -> bool:
    """Whether any quarantined column name appears anywhere inside ``value``.

    Whole-token match against the JSON text. ``F3891`` must not be flagged
    because ``F389`` is a prefix of it, so the boundary check is on the
    surrounding characters rather than a substring test.
    """
    blob = json.dumps(value, default=str)
    return any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])",
                         blob)
               for tok in _quarantined_tokens())


def quarantined_reason_columns(reasons: list[dict[str, Any]] | None,
                               registry: dict[str, Any] | None = None,
                               ) -> list[str]:
    """Which of these reason rows name a column the firewall quarantines."""
    bad: list[str] = []
    for r in reasons or []:
        col = r.get("feature")
        if not col:
            continue
        try:
            firewall.assert_clean([str(col)], context="stored evidence",
                                  registry=registry)
        except firewall.LeakageViolation:
            bad.append(str(col))
    return bad


def quarantined_columns_anywhere(score: dict[str, Any]) -> list[str]:
    """Every quarantined column named anywhere in this payload.

    The complement of :func:`quarantined_reason_columns`, which reads only
    ``top_reasons`` because that is where SHAP reasoning lives. This one makes
    no assumption about shape: it is the check that would still have caught the
    2026-08-26 incident if the retired bundle had written its drivers under some
    other key. A payload is a document, not a schema, and the guarantee has to
    hold over the document.
    """
    blob = json.dumps(score, default=str)
    return sorted(
        tok for tok in _quarantined_tokens()
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", blob))


def retired_detail(case_id: str, prov: dict[str, Any]) -> dict[str, Any]:
    """The 409 body for a payload that is no longer the production model's."""
    stored = prov["stored_model_version"]
    current = prov["current_model_version"]
    return {
        "error": RETIRED_ERROR,
        "case_id": case_id,
        **prov,
        "message": (
            f"This case was scored by model_version {stored!r}, which is no "
            f"longer the production model (now {current!r}). Its stored "
            "explanation was produced under a different feature policy and is "
            "retained as an audit record, not as current evidence. Re-score the "
            "account to obtain current evidence; see "
            f"/v1/proofgraph/{case_id}/provenance for what it was built from."),
    }


def inadmissible_detail(case_id: str, prov: dict[str, Any],
                        detail: str) -> dict[str, Any]:
    """The 409 body for a current payload that named a quarantined column."""
    return {
        "error": INADMISSIBLE_ERROR,
        "case_id": case_id,
        **prov,
        "message": ("A quarantined feature reached this evidence and it was "
                    f"refused rather than shown: {detail}"),
    }


def assert_servable_as_current_evidence(
        case_id: str, score: dict[str, Any],
        registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gate a stored payload before any of it is rendered as a finding.

    Returns the provenance block on success so the caller can echo it; raises
    409 otherwise. Callers that must stay reachable for a retired case - the
    case file itself, which an analyst still needs for its action history -
    should call :func:`evidence_status` instead and redact rather than refuse.
    """
    prov = provenance(score)
    if prov["status"] != PROVENANCE_CURRENT:
        log.info("refused retired evidence for case %s (stored=%s current=%s)",
                 case_id, prov["stored_model_version"],
                 prov["current_model_version"])
        raise HTTPException(409, retired_detail(case_id, prov))
    bad = sorted(set(quarantined_reason_columns(score.get("top_reasons"), registry))
                 | set(quarantined_columns_anywhere(score)))
    if bad:
        log.error("QUARANTINE BREACH: current payload for case %s names %s",
                  case_id, bad)
        raise HTTPException(
            409, inadmissible_detail(case_id, prov, ", ".join(bad)))
    return prov


def evidence_status(case_id: str, score: dict[str, Any] | None,
                    registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Non-raising form: say what this payload is, and why, without refusing.

    The case file needs this. An analyst looking at a case scored by a retired
    model still has a legitimate reason to open it - the action history, the
    feedback and the audit trail are all still theirs - and 404-ing the whole
    page to protect them from two columns would be the wrong trade. So the page
    stays reachable and the *evidence* is what gets withheld, labelled with the
    reason it was withheld.
    """
    if score is None:
        return {"admissible_as_current_evidence": False,
                "reason": "NO_STORED_PAYLOAD",
                "explanation": "this case has no stored score payload",
                "provenance": None,
                "quarantined_features_used": []}
    prov = provenance(score)
    bad = sorted(set(quarantined_reason_columns(score.get("top_reasons"), registry))
                 | set(quarantined_columns_anywhere(score)))
    if prov["status"] != PROVENANCE_CURRENT:
        return {
            "admissible_as_current_evidence": False,
            "reason": RETIRED_ERROR,
            "explanation": retired_detail(case_id, prov)["message"],
            "provenance": prov,
            "quarantined_features_used": bad,
        }
    if bad:
        return {
            "admissible_as_current_evidence": False,
            "reason": INADMISSIBLE_ERROR,
            "explanation": inadmissible_detail(
                case_id, prov, ", ".join(bad))["message"],
            "provenance": prov,
            "quarantined_features_used": bad,
        }
    return {"admissible_as_current_evidence": True, "reason": None,
            "explanation": None, "provenance": prov,
            "quarantined_features_used": []}


def redact_retired_evidence(case_id: str, score: dict[str, Any],
                            status: dict[str, Any]) -> dict[str, Any]:
    """A copy of the payload with the inadmissible parts removed, not blanked.

    The keys are *dropped* rather than set to an empty list. A consumer that
    finds ``top_reasons: []`` concludes the model had nothing to say; a consumer
    that finds no key at all has to ask why, and the ``evidence_status`` block
    travelling beside it answers. The score, tier and uncertainty fields stay:
    they are what the retired model reported and the case file is entitled to
    show them, labelled, as history.

    Two passes, and the second is the one that matters. Dropping a fixed list of
    key names is only correct for the payload shape that exists on the day the
    list is written - the incident this module exists for was itself a stored
    shape outliving the code that understood it. So after the named keys go, the
    remaining payload is swept for any quarantined column name and anything
    still carrying one is dropped too, and reported in ``withheld_keys``. A key
    added to the score payload next year inherits the guarantee without anyone
    remembering to come back here.

    The ``evidence_withheld`` notice is built after the sweep and is exempt from
    it: naming which columns were withheld is the audit trail, not a case
    against the account.
    """
    if status.get("admissible_as_current_evidence"):
        return score
    withheld = [k for k in _EVIDENCE_KEYS if k in score]
    redacted = {k: v for k, v in score.items() if k not in _EVIDENCE_KEYS}
    for key in [k for k, v in redacted.items()
                if _names_a_quarantined_column(v)]:
        log.warning("case %s: payload key %r still named a quarantined column "
                    "after the known evidence keys were dropped; withholding it "
                    "as well", case_id, key)
        redacted.pop(key)
        withheld.append(key)
    redacted["evidence_withheld"] = {
        "reason": status["reason"],
        "explanation": status["explanation"],
        "withheld_keys": withheld,
        "quarantined_features_used": status["quarantined_features_used"],
        "audit_record_available_at": f"/v1/proofgraph/{case_id}/provenance",
    }
    return redacted
