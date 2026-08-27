# Final Security Report

Live evidence: `artifacts/testing/security_results.json` +
`artifacts/testing/no_mcp_scan.txt`; standing controls in
`docs/SECURITY_AND_PRIVACY_REPORT.md`. Regenerate: `bash scripts/test_security.sh`.

## Tested and passing

| Area | Check | Result |
|---|---|---|
| Secrets | `.env` untracked; example committed; regex scan for hardcoded keys; log redaction unit-tested | PASS |
| Model artifacts | bundle SHA-256 recomputed == manifest; loaded only from repo-owned path; uploads never accepted as models | PASS |
| Deserialisation | joblib loads restricted to pipeline-produced artifacts; no user-supplied pickle paths | PASS |
| Path traversal | no user-supplied filesystem paths; report/case ids are server-generated | PASS |
| CSV formula injection | `csv_safe` prefixes `= + - @`; asserted on live batch output | PASS |
| Upload abuse | 512 MB cap, MAX_UPLOAD_MB (413), extension/MIME allowlist, corrupted-file 422 without crash | PASS |
| SQL injection | parameterised queries throughout; hostile case-id path returns 404 with service healthy | PASS |
| XSS | API is JSON-only; React text nodes auto-escape analyst notes | PASS |
| Malformed JSON | 422 | PASS |
| Audit integrity | `audit_events` UPDATE/DELETE blocked by SQLite triggers (tamper attempt raises) | PASS |
| High-impact actions | freeze recommendation requires actor + reason + second approver; nothing punitive is automatic | PASS |
| PII | masked account references enforced by regex; dataset already anonymised; no external calls | PASS |
| CORS | allowlist = local dev frontend only | PASS |
| Dependencies | pinned in `requirements.lock`; `pip-audit` recommended in CI (documented) | PASS (advisory) |

## Residual risk (stated)

- RBAC is actor-string based for the demo (full identity/SSO integration is a
  production extension).
- SQLite for demo; PostgreSQL + TLS + at-rest encryption documented for
  production.
