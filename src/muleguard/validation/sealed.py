"""Sealed Validation Protocol (addendum UPDATE 12).

The problem this solves is one of credibility, not accuracy. When a team
uploads a labelled validation file, scores it, and reports a number, nobody can
tell from the artifact whether the predictions were produced before or after
somebody looked at the labels. "We didn't peek" is unfalsifiable.

The seal makes it falsifiable. The order is fixed and enforced in code:

    1. the target column is located and **withheld** from the scoring frame,
    2. predictions are produced from the remaining columns,
    3. predictions are written to disk,
    4. a SHA-256 of that file is computed,
    5. a manifest recording the hash, the UTC timestamp, the model version and
       the SHA-256 of the *input* is written,
    6. only then may labels be read.

:func:`reveal_metrics` recomputes the prediction hash before it will score
anything. If the predictions were edited after sealing - by anyone, for any
reason - the hash no longer matches and the reveal is refused. The metrics a
judge sees therefore provably belong to predictions that existed before the
labels were opened.

Nothing here trains, tunes or calibrates. Uploaded rows never re-enter the
model (addendum UPDATE 3).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("validation.sealed")

SEAL_DIR = settings.ARTIFACTS_DIR / "sealed_validation"

STATE_SEALED = "PREDICTIONS_SEALED"
STATE_REVEALED = "METRICS_REVEALED"
STATE_BROKEN = "SEAL_BROKEN"


class SealBroken(RuntimeError):
    """Raised when sealed predictions no longer match their recorded hash."""


class BadSealId(ValueError):
    """Raised when a seal id is not a plain identifier this module issued."""


# Seal ids are minted here and nowhere else, in exactly one shape, so a strict
# allow-list costs nothing. It matters because the id becomes a filename: an id
# of "../../config" would resolve to a .json outside SEAL_DIR, and the reveal
# path writes the manifest back, which would make that an arbitrary file write.
_SEAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Windows resolves these to devices whatever the extension, so "CON.json" is
# the console, not a file in this directory. Reading one blocks; writing one
# silently discards the manifest. This is the reason the containment check
# above is not sufficient on its own - the path *is* inside SEAL_DIR.
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def safe_seal_id(seal_id: str) -> str:
    """Return ``seal_id`` if it can only ever name a file inside SEAL_DIR."""
    if (not isinstance(seal_id, str) or not _SEAL_ID.match(seal_id)
            or ".." in seal_id
            or seal_id.split(".")[0].upper() in _RESERVED):
        raise BadSealId(f"not a valid seal id: {seal_id!r}")
    return seal_id


def seal_path(seal_id: str, suffix: str = ".json") -> Path:
    """Resolve a seal artifact path, refusing anything that escapes SEAL_DIR.

    The regex above already rejects separators, so the containment check is
    belt and braces - but it is the check that stays correct if the regex is
    ever loosened, and on Windows it also catches the reserved device names
    that look harmless to a character class.
    """
    p = (SEAL_DIR / f"{safe_seal_id(seal_id)}{suffix}").resolve()
    if not str(p).startswith(str(SEAL_DIR.resolve())):
        raise BadSealId(f"seal id escapes the seal directory: {seal_id!r}")
    return p


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_fingerprint(df: pl.DataFrame) -> str:
    """Content hash of a frame: shape, column names, and every cell.

    Uses the CSV serialisation rather than the in-memory buffers so the value
    is reproducible across polars versions and across machines - the property
    that makes it worth printing next to a metric.
    """
    head = f"{df.height}x{df.width}|" + "|".join(df.columns)
    body = df.write_csv().encode("utf-8")
    return _sha256_bytes(head.encode("utf-8") + b"\n" + body)


@dataclass
class SealRecord:
    """The tamper-evident manifest for one sealed prediction run."""

    seal_id: str
    sealed_utc: str
    model_version: str
    n_rows: int
    input_sha256: str
    prediction_sha256: str
    prediction_path: str
    score_column: str
    target_column_detected: str | None
    target_withheld: bool
    state: str = STATE_SEALED
    revealed_utc: str | None = None
    metrics: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def manifest_path(self) -> Path:
        return seal_path(self.seal_id)

    def save(self) -> Path:
        SEAL_DIR.mkdir(parents=True, exist_ok=True)
        p = self.manifest_path
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def load_seal(seal_id: str) -> SealRecord:
    p = seal_path(seal_id)
    if not p.exists():
        raise FileNotFoundError(f"no sealed run {seal_id!r}")
    return SealRecord(**json.loads(p.read_text(encoding="utf-8")))


def list_seals() -> list[dict[str, Any]]:
    if not SEAL_DIR.exists():
        return []
    out = []
    for p in sorted(SEAL_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("unreadable seal manifest: %s", p.name)
    return out


def withhold_target(df: pl.DataFrame,
                    target: str | None = None) -> tuple[pl.DataFrame, np.ndarray | None, str | None]:
    """Split labels off the uploaded frame before anything else touches it.

    Returns the frame *without* the target, the label vector held aside, and
    the name of the column that was removed. The labels are returned rather
    than silently dropped because the caller has to hold them somewhere until
    the reveal - but they leave this function as a separate object, so no code
    path downstream of scoring can reach them by accident.
    """
    tgt = target or settings.TARGET_COLUMN
    if tgt not in df.columns:
        return df, None, None
    y = df[tgt].cast(pl.Float64, strict=False).to_numpy()
    return df.drop(tgt), y, tgt


def seal_predictions(
    upload: pl.DataFrame,
    scores: Sequence[float],
    *,
    model_version: str,
    reference: Sequence[str] | None = None,
    target_column: str | None = None,
    extra_columns: dict[str, Sequence[Any]] | None = None,
    seal_id: str | None = None,
) -> SealRecord:
    """Write predictions, hash them, and record the manifest.

    ``upload`` is the frame as received (labels may still be present); the
    fingerprint is taken of it so the manifest states exactly which file was
    scored. ``scores`` must already have been produced from a target-free
    frame - :func:`withhold_target` is how that is done, and the record notes
    whether a target column was present in the input at all.
    """
    if len(scores) != upload.height:
        raise ValueError(
            f"{len(scores)} scores for {upload.height} uploaded rows")
    SEAL_DIR.mkdir(parents=True, exist_ok=True)
    sid = safe_seal_id(
        seal_id or f"seal-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}")
    tgt = target_column or settings.TARGET_COLUMN
    detected = tgt if tgt in upload.columns else None

    cols: dict[str, Any] = {
        "reference": (list(reference) if reference is not None
                      else [f"ROW-{i}" for i in range(upload.height)]),
        "risk_score": [float(s) for s in scores],
    }
    for k, v in (extra_columns or {}).items():
        if len(v) != upload.height:
            raise ValueError(f"extra column {k!r} has the wrong length")
        cols[k] = list(v)
    preds = pl.DataFrame(cols)

    path = seal_path(sid, "_predictions.csv")
    preds.write_csv(path)

    rec = SealRecord(
        seal_id=sid,
        sealed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        model_version=str(model_version),
        n_rows=upload.height,
        input_sha256=frame_fingerprint(upload),
        prediction_sha256=_sha256_file(path),
        prediction_path=str(path),
        score_column="risk_score",
        target_column_detected=detected,
        target_withheld=detected is not None,
        notes=[
            "predictions were produced from a frame with the target column removed",
            "no label was read before this manifest was written",
            "the uploaded rows were not used to retrain or recalibrate any model",
        ],
    )
    rec.save()
    log.info("sealed %s: %d rows, pred sha %s, target %s",
             sid, rec.n_rows, rec.prediction_sha256[:12],
             "withheld" if rec.target_withheld else "absent from upload")
    return rec


def verify_seal(rec: SealRecord) -> tuple[bool, str]:
    """Recompute the prediction hash and compare it to the manifest."""
    p = Path(rec.prediction_path)
    if not p.exists():
        return False, f"prediction file missing: {p}"
    actual = _sha256_file(p)
    if actual != rec.prediction_sha256:
        return False, (f"prediction file changed since sealing "
                       f"(manifest {rec.prediction_sha256[:12]}, "
                       f"file {actual[:12]})")
    return True, "prediction file matches the sealed hash"


def reveal_metrics(seal_id: str, y_true: Sequence[float],
                   *, budgets: Sequence[int] = (25, 50, 100)) -> dict[str, Any]:
    """Verify the seal, then - and only then - score against the labels.

    The verification is not a formality: if it fails this raises rather than
    reporting a number, because a metric computed from predictions that changed
    after sealing means nothing.
    """
    from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                                 roc_auc_score)

    rec = load_seal(seal_id)
    ok, why = verify_seal(rec)
    if not ok:
        rec.state = STATE_BROKEN
        rec.notes.append(f"verification failed at reveal: {why}")
        rec.save()
        raise SealBroken(why)

    preds = pl.read_csv(rec.prediction_path)
    s = preds[rec.score_column].to_numpy()
    y = np.asarray(y_true, dtype=float)
    if len(y) != len(s):
        raise ValueError(f"{len(y)} labels for {len(s)} sealed predictions")
    finite = ~np.isnan(y)
    y, s = y[finite].astype(int), s[finite]

    n_pos = int(y.sum())
    metrics: dict[str, Any] = {
        "n_rows_scored": int(len(y)),
        "n_positives": n_pos,
        "prevalence": float(y.mean()) if len(y) else 0.0,
    }
    if n_pos == 0 or n_pos == len(y):
        metrics["status"] = "SINGLE_CLASS_ONLY"
        metrics["note"] = ("the revealed labels contain one class only; "
                           "ranking metrics are undefined")
    else:
        order = np.argsort(-s, kind="stable")
        metrics.update({
            "status": "OK",
            "pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "lift_over_prevalence": float(
                average_precision_score(y, s) / max(float(y.mean()), 1e-12)),
            "recall_at_budget": {
                str(k): float(y[order[:min(k, len(y))]].sum() / n_pos)
                for k in budgets
            },
            "precision_at_budget": {
                str(k): float(y[order[:min(k, len(y))]].mean())
                for k in budgets
            },
            "mcc_at_top_100": float(matthews_corrcoef(
                y, np.isin(np.arange(len(y)), order[:min(100, len(y))]).astype(int))),
        })

    rec.state = STATE_REVEALED
    rec.revealed_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    rec.metrics = metrics
    rec.notes.append("labels were opened only after the hash above was verified")
    rec.save()
    log.info("revealed %s: %s", seal_id, metrics.get("status"))
    return {
        "seal": rec.to_dict(),
        "verification": {"verified": True, "detail": why},
        "metrics": metrics,
        "integrity_statement": (
            f"Predictions were sealed at {rec.sealed_utc} with SHA-256 "
            f"{rec.prediction_sha256}. The file was re-hashed at reveal time "
            "and matched. These metrics therefore belong to predictions that "
            "existed before any label was read."
        ),
    }
