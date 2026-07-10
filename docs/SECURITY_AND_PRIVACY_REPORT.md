# Security and Privacy Report

## Data protection

| Control | Implementation |
|---|---|
| Raw data immutability | original XLSX never written; read-only copy in `data/raw/`; SHA-256 verified by tests and release gate |
| Raw data not in git | `.gitignore` blocks `DataSet.xlsx`, `data/raw|interim|processed`; verified by `tests/security` |
| PII | dataset is already anonymised (Fxxxx). API accepts only masked account references (`^[A-Za-z0-9_-]{1,64}$`); free-text PII-looking refs are rejected |
| No external calls | scoring is fully local; the optional LLM is localhost Ollama; no telemetry |
| Secrets | `.env` untracked (`.env.example` committed); log formatter redacts `api_key/secret/password/token`; no hardcoded secrets (regex-scanned in tests) |

## Application security

| Control | Implementation |
|---|---|
| Input validation | Pydantic schemas; batch limited to 500; request size bounded; target column rejected in requests |
| Silent-fill prevention | missing selected features → `422 SCHEMA_ERROR`, never zero-filled |
| Rate limiting | 240 req/min per client (demo-scale) |
| CSV/formula injection | `csv_safe()` prefixes `= + - @` cells; unit-tested |
| Path traversal | no user-supplied paths are opened; reports are stored by server-generated case ids |
| Model artifacts | joblib bundle hashed (SHA-256 in manifest + registry); loaded only from the repo-owned artifacts directory, never from uploads |
| Pickle caution | joblib deserialisation happens only on files produced by this pipeline in-place; upload endpoints never accept model files |
| Audit trail | `audit_events` is append-only — enforced by SQLite triggers (UPDATE/DELETE abort) and covered by an integration test |
| High-impact actions | `RECOMMEND_FREEZE` requires a second authorised approver id + reason; nothing punitive is automatic |
| Role separation | analyst actor ids recorded on every action; approver distinct from actor for freeze recommendations (full RBAC is a production extension, stated honestly) |

## Dependency posture

Versions pinned in `requirements.lock` (pip freeze of the build venv).
`pip-audit`/Dependabot recommended in CI for production (documented extension).

## Production guidance (not demo-implemented)

- PostgreSQL with TLS + at-rest encryption instead of SQLite
- retention: scores/cases per bank policy (suggested 7 years for audit events,
  aligned with Indian banking record-keeping norms — subject to bank counsel)
- KMS-managed secrets; mTLS between services; SIEM export of audit events
