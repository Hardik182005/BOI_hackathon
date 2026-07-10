# Final End-to-End Scenario Report

Live evidence: `artifacts/testing/e2e_results.json` (harness re-executes
scenarios against the running stack) + `artifacts/evidence/demo_scenarios.json`
(full per-scenario payloads, DEV-set rows only).

| Scenario | Expected | Measured outcome |
|---|---|---|
| A — High-risk mule-like account | high calibrated risk, critical/urgent review, reasons, human required, no auto-freeze | labelled mule scored calibrated risk ≈1.00 → **CRITICAL_REVIEW**, `HUMAN_REVIEW_REQUIRED`, SHAP drivers attached, `auto_action=null` |
| B — Legitimate business look-alike | elevated first-pass risk, protection layer routes to review, no criminal label | screener risk 0.96 but hard-negative verifier disagreed → capped to **URGENT_REVIEW** with explicit "look-alike protection" reason |
| C — Low-current-risk account | MONITOR, "not currently flagged", never "safe" | **MONITOR**, `NOT_CURRENTLY_FLAGGED`, limitations attached |
| D — OOD account (synthetic, labelled) | OOD_REVIEW, no confident normal result | extreme values → **OOD_REVIEW**, "score not trusted for this input" |
| E — Ollama offline | same score/tier, deterministic explanation, no broken page | narrator falls back deterministically; scoring identical; UI shows source badge |
| F — Ollama hallucination | rejected + fallback + audit | planted output tripped 7 rejection rules (score, tier, F999, ₹ amount, guilt, disallowed action, missing limitations); deterministic fallback used |
| G — Invalid batch upload | safe rejection, clear error, no crash | corrupted XLSX → 422 with parse error; `/health/live` still 200 |
| H — Model restart | same version, same predictions | bundle SHA-256 served by API == manifest; save/load determinism pytest |
| I — Analyst decision | explicit action, actor+timestamp stored, no auto bank action | freeze recommendation without approver → 422; with actor+reason → recorded in case_actions + append-only audit |
| J — Drift warning | monitoring alert, no auto promotion | drift endpoint serves PSI status; champion/challenger promotion is a recorded human decision (registry), no auto path exists |

All scenario rows come from the development split (the locked test is never
used for demos); the OOD case is an explicitly labelled synthetic
perturbation of a real dev account.
