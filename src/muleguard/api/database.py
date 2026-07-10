"""SQLite persistence with an append-only audit trail.

Critical audit events are append-only by construction: no UPDATE/DELETE code
paths exist for audit_events, and a trigger blocks them at the engine level.
PostgreSQL support = swap the connection string via MULEGUARD_DB env var
(schema uses portable SQL).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from muleguard import settings
from muleguard.logging import get_logger

log = get_logger("api.database")

DB_PATH = Path(os.environ.get("MULEGUARD_DB", settings.ARTIFACTS_DIR / "muleguard.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    bundle_sha256 TEXT NOT NULL,
    winner TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'champion'
);
CREATE TABLE IF NOT EXISTS scoring_requests (
    request_id TEXT PRIMARY KEY,
    correlation_id TEXT,
    account_reference TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES scoring_requests(request_id),
    account_reference TEXT NOT NULL,
    calibrated_risk REAL NOT NULL CHECK (calibrated_risk >= 0 AND calibrated_risk <= 1),
    risk_tier TEXT NOT NULL,
    model_agreement REAL,
    conformal_status TEXT,
    ood_status TEXT,
    anomaly_percentile REAL,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    account_reference TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    calibrated_risk REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    assignee TEXT,
    score_id INTEGER REFERENCES scores(id),
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    approved_by TEXT,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyst_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    actor TEXT NOT NULL,
    verdict TEXT NOT NULL,
    notes TEXT,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS drift_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generated_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    correlation_id TEXT,
    before_state TEXT,
    after_state TEXT,
    model_version TEXT,
    detail_json TEXT,
    created_utc TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_events_no_update
BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
"""


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def audit(event_type: str, actor: str, *, correlation_id: str | None = None,
          before: Any = None, after: Any = None, model_version: str | None = None,
          detail: Any = None) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO audit_events (event_type, actor, correlation_id, before_state,"
            " after_state, model_version, detail_json, created_utc)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (event_type, actor, correlation_id,
             json.dumps(before) if before is not None else None,
             json.dumps(after) if after is not None else None,
             model_version,
             json.dumps(detail) if detail is not None else None,
             utcnow()),
        )


def record_score(account_reference: str, result: dict[str, Any],
                 correlation_id: str | None = None) -> tuple[str, int, str | None]:
    """Persist request+score; open a case for review tiers. Returns
    (request_id, score_id, case_id)."""
    request_id = str(uuid.uuid4())
    now = utcnow()
    with connect() as c:
        c.execute(
            "INSERT INTO scoring_requests (request_id, correlation_id, account_reference,"
            " model_version, created_utc) VALUES (?,?,?,?,?)",
            (request_id, correlation_id, account_reference, result["model_version"], now),
        )
        cur = c.execute(
            "INSERT INTO scores (request_id, account_reference, calibrated_risk, risk_tier,"
            " model_agreement, conformal_status, ood_status, anomaly_percentile,"
            " payload_json, created_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (request_id, account_reference, result["calibrated_risk"], result["risk_tier"],
             result["model_agreement"], result["conformal_status"], result["ood_status"],
             result["anomaly_percentile"], json.dumps(result), now),
        )
        score_id = cur.lastrowid
        case_id = None
        if result["risk_tier"] != "MONITOR":
            case_id = f"CASE-{uuid.uuid4().hex[:10].upper()}"
            c.execute(
                "INSERT INTO cases (case_id, account_reference, risk_tier, calibrated_risk,"
                " status, score_id, created_utc, updated_utc) VALUES (?,?,?,?,?,?,?,?)",
                (case_id, account_reference, result["risk_tier"],
                 result["calibrated_risk"], "OPEN", score_id, now, now),
            )
    audit("SCORE", "system", correlation_id=correlation_id,
          after={"account": account_reference, "tier": result["risk_tier"],
                 "risk": result["calibrated_risk"], "case_id": case_id},
          model_version=result["model_version"])
    return request_id, score_id, case_id
