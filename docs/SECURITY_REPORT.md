# Security Report

Master prompt §31. Controls, tests, and honestly-stated residual risk.

Live evidence: `artifacts/testing/security_results.json`,
`artifacts/testing/no_mcp_scan.txt`. Regenerate:
`bash scripts/test_security.sh`. Release-gate check `attack_surface_covered`
reports **49 security tests** covering SQL injection, XSS, path traversal and CSV
injection.

---

## 1. Data protection

| Control | Implementation |
|---|---|
| **Raw workbooks immutable** | the original XLSX is never written; a read-only copy lives in `data/raw/`; SHA-256 verified by tests **and** by the release gate (`raw_data_unmodified` PASS) |
| Raw data not committed | `.gitignore` blocks `DataSet.xlsx` and `data/raw\|interim\|processed`; verified by a security test, not by convention |
| PII | the dataset is already anonymised (`Fxxxx`). The API accepts only masked account references, regex `^[A-Za-z0-9_-]{1,64}$`; free-text PII-looking references are rejected |
| No raw PII in logs | the log formatter redacts `api_key` / `secret` / `password` / `token`; account references are masked before they reach a log line |
| **No external calls** | scoring is fully local. The optional LLM is localhost Ollama. No telemetry, no API keys, no internet |
| Secrets | `.env` untracked (`.env.example` committed); no hardcoded secrets — regex-scanned in tests; gate check `no_secrets_committed` PASS |

---

## 2. Application security

| Control | Implementation |
|---|---|
| Input validation | Pydantic schemas throughout; batch limited to 500 rows; request size bounded; the **target column is rejected** if present in a scoring request |
| **Silent-fill prevention** | a missing required feature returns `422 SCHEMA_ERROR` — it is **never** zero-filled. A confident score for an account the model has no information about is the most dangerous failure this system could have |
| Rate limiting | 240 req/min per client (demo-scale guardrail) |
| CSV / formula injection | `csv_safe()` prefixes cells beginning `=`, `+`, `-`, `@`; unit-tested and asserted on live batch output. Without it, an exported alert list opened in Excel is a code-execution vector |
| Path traversal | no user-supplied filesystem path is ever opened. Report and case ids are server-generated. **Seal ids** — which do become filenames — are validated against `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` |
| Windows reserved device names | `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9` are rejected separately, because containment is not enough: `CON.json` resolves to the console *while sitting inside* the seal directory — reading it blocks, writing it silently discards data |
| SQL injection | parameterised queries throughout; a hostile case-id path returns 404 with the service healthy |
| XSS | the API is JSON-only; React text nodes auto-escape analyst notes |
| Upload abuse | 512 MB cap, MAX_UPLOAD_MB (413), extension and MIME allowlist, corrupted file → 422 without a crash |
| Malformed JSON | 422 |
| **Model artefacts** | the joblib bundle is SHA-256-hashed in the manifest and registry, and loaded **only** from the repo-owned artifacts directory. Upload endpoints never accept model files |
| Deserialisation | joblib loads are restricted to pipeline-produced artefacts; no user-supplied pickle path exists |
| **Audit trail** | `audit_events` is append-only, enforced by SQLite triggers — `UPDATE` and `DELETE` abort. Covered by an integration test that attempts tampering and asserts the raise |
| High-impact actions | `RECOMMEND_FREEZE` requires a second authorised approver id **and** a reason. **Nothing punitive is automatic** — gate check `no_auto_freeze_paths` PASS |
| Role separation | analyst actor ids are recorded on every action; the approver must differ from the actor |
| CORS | allowlist is the local dev frontend only |

---

## 3. Model-integrity attacks

This system's specific threat is not only "can someone break in" but **"can
anyone change a decision without it being visible"**.

| Attack | Defence | Verified |
|---|---|---|
| LLM alters a score | no code path from narrator to score; 8 validator rules | gate `llm_cannot_alter_score` PASS |
| A hallucinated narrative is served as evidence | planted hallucination in the demo pipeline | rejected for 8 reasons |
| Uploaded data retrains or shifts the model | no fit call on the upload path; bundle fingerprint checked before and after | `afd0dc1d8fc02eb9` byte-identical across all 8 dry-run variants |
| Predictions edited after the labels are seen | Sealed Validation Protocol — SHA-256 recomputed at reveal; mismatch → `SEAL_BROKEN`, reveal refused | `tests/unit/test_sealed_validation.py` |
| A quarantined feature re-enters the model | firewall + gate check | `no_target_or_f3912_leakage` PASS — 120 bundle features disjoint from 13 quarantined |
| Fabricated graph edges | adapter contract forbids derived edges | gate `graph_never_fabricates_edges` PASS |
| Accusatory vocabulary reaches an analyst | `assert_language_safe` on every serialised payload + a source scan | gate `no_forbidden_verdict_vocabulary` PASS |

---

## 4. No external agent dependency

`artifacts/testing/no_mcp_scan.txt` records a scan of shipped source: **no MCP
client, no browser-agent dependency, no external LLM API key, no internet
requirement for scoring.** The product runs on a disconnected laptop.

---

## 5. Dependency posture

Versions are pinned in `requirements.lock` (a pip freeze of the build venv).
`pip-audit` / Dependabot in CI is **recommended and documented, not implemented**
— marked PASS (advisory) rather than PASS.

---

## 6. Residual risk — stated, not buried

| Risk | Status |
|---|---|
| **RBAC is actor-string based** | adequate for a demo; full identity/SSO integration is a production extension. Anyone who can call the API can claim to be any analyst |
| **SQLite** | fine for a single-node demo; PostgreSQL with TLS and at-rest encryption is documented for production |
| **No dependency CVE scanning in CI** | recommended, not wired up |
| **No secrets manager** | `.env` for the demo; KMS-managed secrets documented for production |
| **Single-node, no mTLS** | service-to-service TLS and SIEM export of audit events are production guidance |

## 7. Production guidance (documented, not implemented)

- PostgreSQL with TLS and at-rest encryption in place of SQLite
- retention policy for scores and cases per bank policy — 7 years suggested for
  audit events, aligned with Indian banking record-keeping norms, subject to bank
  counsel
- KMS-managed secrets, mTLS between services, SIEM export of the audit stream
- full RBAC with SSO, replacing actor strings

Each of these is a known gap with a named remedy. None of them is claimed as
present.
