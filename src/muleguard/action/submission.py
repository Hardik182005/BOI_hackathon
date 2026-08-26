"""Competition and analyst export builders.

This module reshapes frozen predictions into the two files a submission needs:
a minimal ``row_id,F3924`` file for the organiser, and a richer analyst file
for human reviewers. Its shape comes entirely from
``configs/submission_format.yaml`` because the organiser's required format is
not knowable in advance.

The one invariant that matters here: **nothing in this module can change a
score.** It reads calibrated risks that were produced upstream and formats
them. It never fits, recalibrates, re-thresholds or reorders. Every export is
validated before it is written, because a silently malformed submission is
worse than no submission at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from ..features import firewall
from ..settings import TARGET_COLUMN, load_config
from ..utils import sha256_file

__all__ = [
    "ExportValidationError",
    "SubmissionFormat",
    "resolve_row_ids",
    "build_minimal_submission",
    "build_analyst_export",
    "validate_minimal_submission",
    "write_exports",
]


class ExportValidationError(ValueError):
    """An export failed its pre-write checks and was not written."""


def _quarantined_features() -> set[str]:
    """Feature names that may never appear in an export.

    Read from the Feature Availability Firewall, not from
    ``configs/leakage_quarantine.yaml``. That file predates the firewall and
    still lists **4** columns; the firewall quarantines **13**. An export built
    against the short list could therefore have carried nine quarantined
    columns - including ``F3898 MIN_RESOLVE_DAYS`` and ``F3914 FALSE_POSITIVE``,
    the two that surfaced as prosecution evidence on 2026-08-26 - into a file
    handed to an analyst or an organiser. The older file is kept as the record
    of the original four leak findings and is unioned in, so a name that only
    ever appeared there is still refused.
    """
    cfg = firewall.config()
    names = set(cfg.hard_quarantine) | set(cfg.conditional_quarantine)         | set(cfg.fairness_excluded)
    try:
        legacy = load_config("leakage_quarantine")
    except FileNotFoundError:
        legacy = {}
    names |= {str(e["feature"]) for e in legacy.get("quarantine", [])
              if e.get("feature")}
    names.add(TARGET_COLUMN)
    return names


class SubmissionFormat:
    """The submission shape, loaded from ``configs/submission_format.yaml``."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self._cfg = cfg if cfg is not None else load_config("submission_format")
        self.version: str = str(self._cfg.get("submission_format_version", "1.0"))
        self.minimal: dict[str, Any] = dict(self._cfg.get("minimal", {}))
        self.analyst: dict[str, Any] = dict(self._cfg.get("analyst", {}))
        self.validation: dict[str, Any] = dict(self._cfg.get("validation", {}))

    # -- minimal ---------------------------------------------------------
    @property
    def id_column(self) -> str:
        return str(self.minimal.get("id_column", "row_id"))

    @property
    def prediction_column(self) -> str:
        return str(self.minimal.get("prediction_column", TARGET_COLUMN))

    @property
    def prediction_type(self) -> str:
        t = str(self.minimal.get("prediction_type", "probability")).lower()
        if t not in {"probability", "binary"}:
            raise ExportValidationError(
                f"prediction_type must be 'probability' or 'binary', got {t!r}")
        return t

    @property
    def decimals(self) -> int:
        return int(self.minimal.get("probability_decimals", 6))

    @property
    def id_candidates(self) -> list[str]:
        return [str(c) for c in self.minimal.get("id_column_candidates", [])]

    def binary_threshold(self, policy_threshold: float | None) -> float:
        """Resolve the 0/1 cutoff. ``policy`` defers to the frozen snapshot."""
        raw = self.minimal.get("binary_threshold", "policy")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        if str(raw).lower() == "policy":
            if policy_threshold is None:
                raise ExportValidationError(
                    "binary_threshold is 'policy' but no frozen policy threshold "
                    "was supplied; refusing to invent a cutoff")
            return float(policy_threshold)
        raise ExportValidationError(f"unusable binary_threshold {raw!r}")

    def flag(self, name: str, default: bool = True) -> bool:
        return bool(self.validation.get(name, default))


def resolve_row_ids(frame: pl.DataFrame, fmt: SubmissionFormat) -> tuple[list, str]:
    """Return ``(ids, source)`` for the upload.

    A caller-supplied identifier column is preferred so the organiser can join
    on their own key; otherwise ids are positional ``1..N``. Row order is
    never changed - positional ids are only meaningful if order is preserved.
    """
    for cand in fmt.id_candidates:
        if cand in frame.columns:
            col = frame[cand]
            if col.null_count() == 0 and col.n_unique() == frame.height:
                return col.to_list(), f"upload_column:{cand}"
    return list(range(1, frame.height + 1)), "positional"


def _rounded(scores: Sequence[float], decimals: int) -> list[float]:
    return [round(float(s), decimals) for s in scores]


def build_minimal_submission(
    scores: Sequence[float],
    *,
    row_ids: Sequence[Any] | None = None,
    n_rows: int | None = None,
    fmt: SubmissionFormat | None = None,
    policy_threshold: float | None = None,
) -> pl.DataFrame:
    """Build the two-column competition file.

    ``scores`` are calibrated risks in upload order. They are formatted, not
    recomputed.
    """
    fmt = fmt or SubmissionFormat()
    n = n_rows if n_rows is not None else len(scores)
    if len(scores) != n:
        raise ExportValidationError(
            f"{len(scores)} scores for {n} rows - refusing to export a "
            f"submission whose row count does not match the input")
    ids = list(row_ids) if row_ids is not None else list(range(1, n + 1))
    if len(ids) != n:
        raise ExportValidationError(f"{len(ids)} row ids for {n} rows")

    if fmt.prediction_type == "binary":
        cut = fmt.binary_threshold(policy_threshold)
        preds: list[Any] = [int(float(s) >= cut) for s in scores]
    else:
        preds = _rounded(scores, fmt.decimals)

    return pl.DataFrame({fmt.id_column: ids, fmt.prediction_column: preds})


def _reason_text(reason: dict[str, Any] | None) -> str:
    """Render one SHAP reason code as a short descriptive phrase.

    Descriptive only - it states what the model observed, never what the
    account holder did.
    """
    if not reason:
        return ""
    name = reason.get("verified_semantic_name") or reason.get("feature", "")
    direction = "elevated" if reason.get("direction") == "INCREASES_RISK" else "lowered"
    pct = reason.get("legitimate_percentile")
    if pct is None:
        return f"{name} ({direction} model contribution)"
    return f"{name} at p{float(pct):.0f} of the legitimate cohort ({direction} model contribution)"


def build_analyst_export(
    records: Iterable[dict[str, Any]],
    *,
    row_ids: Sequence[Any] | None = None,
    fmt: SubmissionFormat | None = None,
    policy_threshold: float | None = None,
) -> pl.DataFrame:
    """Build the reviewer-facing file from full scoring records."""
    fmt = fmt or SubmissionFormat()
    recs = list(records)
    n = len(recs)
    ids = list(row_ids) if row_ids is not None else list(range(1, n + 1))
    if len(ids) != n:
        raise ExportValidationError(f"{len(ids)} row ids for {n} scoring records")

    try:
        cut = fmt.binary_threshold(policy_threshold)
    except ExportValidationError:
        cut = None

    k = int(fmt.analyst.get("reason_count", 3))
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(recs):
        risk = float(r.get("calibrated_risk", 0.0))
        reasons = list(r.get("top_reasons") or [])
        row: dict[str, Any] = {
            "row_id": ids[i],
            "calibrated_risk": round(risk, fmt.decimals),
            "binary_prediction": None if cut is None else int(risk >= cut),
            "risk_tier": r.get("risk_tier"),
            "review_status": r.get("decision") or r.get("review_status"),
            "model_agreement": _opt_round(r.get("model_agreement"), 4),
            # Spread across the model committee: high disagreement means the
            # committee is not confident, which is what an analyst needs to see.
            "uncertainty": _opt_round(r.get("prediction_std"), 6),
            "conformal_status": _fmt_conformal(r.get("conformal_status")),
            "ood_status": r.get("ood_status"),
            "model_version": r.get("model_version"),
        }
        for j in range(k):
            row[f"top_reason_{j + 1}"] = _reason_text(
                reasons[j] if j < len(reasons) else None)
        rows.append(row)

    out = pl.DataFrame(rows)
    wanted = [c for c in fmt.analyst.get("columns", out.columns) if c in out.columns]
    extra = [c for c in out.columns if c not in wanted]
    return out.select(wanted + extra)


def _opt_round(v: Any, nd: int) -> float | None:
    return None if v is None else round(float(v), nd)


def _fmt_conformal(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        return "|".join(str(x) for x in sorted(v)) or "EMPTY"
    return str(v)


def validate_minimal_submission(
    df: pl.DataFrame,
    *,
    n_input_rows: int,
    fmt: SubmissionFormat | None = None,
    id_source: str = "positional",
) -> list[str]:
    """Check an export before it is written. Raises on any failure.

    Returns the list of checks that passed, so the caller can record exactly
    what was verified rather than asserting a blanket "validated".
    """
    fmt = fmt or SubmissionFormat()
    passed: list[str] = []
    problems: list[str] = []

    if fmt.flag("require_row_count_match"):
        if df.height != n_input_rows:
            problems.append(
                f"row count {df.height} != {n_input_rows} input rows")
        else:
            passed.append("row_count_matches_input")

    if fmt.flag("forbid_extra_columns"):
        expected = [fmt.id_column, fmt.prediction_column]
        if list(df.columns) != expected:
            problems.append(f"columns {df.columns} != required {expected}")
        else:
            passed.append("exactly_two_required_columns")

    if fmt.flag("require_row_order_preserved") and id_source == "positional":
        if df.height and df[fmt.id_column].to_list() != list(range(1, df.height + 1)):
            problems.append("positional row_id is not a strict 1..N sequence")
        else:
            passed.append("row_order_preserved")

    if fmt.prediction_column in df.columns:
        col = df[fmt.prediction_column]
        if fmt.flag("forbid_null_predictions"):
            if col.null_count():
                problems.append(f"{col.null_count()} null predictions")
            else:
                passed.append("no_null_predictions")
        if fmt.flag("require_probability_range") and fmt.prediction_type == "probability":
            vals = [float(v) for v in col.to_list() if v is not None]
            if vals and (min(vals) < 0.0 or max(vals) > 1.0):
                problems.append(
                    f"probabilities outside [0,1]: min={min(vals)} max={max(vals)}")
            else:
                passed.append("probabilities_in_unit_range")
        if fmt.prediction_type == "binary":
            bad = {v for v in col.to_list() if v not in (0, 1)}
            if bad:
                problems.append(f"binary predictions contain {sorted(bad)[:5]}")
            else:
                passed.append("binary_values_are_0_or_1")

    if fmt.flag("forbid_leakage_columns"):
        quarantined = _quarantined_features()
        # The target name is legal as the prediction column - that is what the
        # organiser asked to be predicted - but never as an echoed input.
        offending = sorted(
            c for c in df.columns
            if c in quarantined and c != fmt.prediction_column)
        if offending:
            problems.append(f"export carries quarantined columns {offending}")
        else:
            passed.append("no_quarantined_columns")

    if problems:
        raise ExportValidationError(
            "submission export failed validation: " + "; ".join(problems))
    return passed


def write_exports(
    out_dir: Path,
    *,
    scores: Sequence[float],
    records: Iterable[dict[str, Any]] | None = None,
    frame: pl.DataFrame | None = None,
    row_ids: Sequence[Any] | None = None,
    id_source: str = "positional",
    fmt: SubmissionFormat | None = None,
    policy_threshold: float | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Write the minimal (and, when records are given, analyst) exports.

    Returns a manifest describing what was written, which checks passed, and
    the SHA-256 of each file so the export can be proven unchanged later.
    """
    fmt = fmt or SubmissionFormat()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if row_ids is None and frame is not None:
        row_ids, id_source = resolve_row_ids(frame, fmt)
    n_rows = frame.height if frame is not None else len(scores)

    minimal = build_minimal_submission(
        scores, row_ids=row_ids, n_rows=n_rows, fmt=fmt,
        policy_threshold=policy_threshold)
    checks = validate_minimal_submission(
        minimal, n_input_rows=n_rows, fmt=fmt, id_source=id_source)

    minimal_path = out_dir / "submission.csv"
    minimal.write_csv(minimal_path)

    manifest: dict[str, Any] = {
        "submission_format_version": fmt.version,
        "model_version": model_version,
        "n_rows": n_rows,
        "id_column": fmt.id_column,
        "id_source": id_source,
        "prediction_column": fmt.prediction_column,
        "prediction_type": fmt.prediction_type,
        "minimal_path": str(minimal_path),
        "minimal_sha256": sha256_file(minimal_path),
        "validation_checks_passed": checks,
        "retraining_performed": False,
        "notes": [
            "the export layer reshapes frozen predictions; it cannot change a score",
            "row order is preserved exactly as uploaded",
        ],
    }

    if records is not None:
        analyst = build_analyst_export(
            records, row_ids=row_ids, fmt=fmt, policy_threshold=policy_threshold)
        analyst_path = out_dir / "analyst_export.csv"
        analyst.write_csv(analyst_path)
        manifest["analyst_path"] = str(analyst_path)
        manifest["analyst_sha256"] = sha256_file(analyst_path)
        manifest["analyst_columns"] = list(analyst.columns)

    (out_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
