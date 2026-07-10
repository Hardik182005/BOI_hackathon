# Ollama Guardrails Report

## Role of the LLM

The local LLM (Ollama) is an **optional narrator**. It converts verified,
structured scoring facts into readable analyst prose. It is architecturally
incapable of affecting a score:

- The scoring path (`muleguard.models.scoring`) contains no LLM call.
- The narrator receives a `NarratorInput` (calibrated risk, tier, agreement,
  conformal/OOD status, SHAP reason facts) — nothing else.
- Its output is parsed against a strict Pydantic schema and a hallucination
  validator; any deviation discards the output entirely in favour of a
  deterministic template (`llm/deterministic_fallback.py`).
- `POST /v1/score` never invokes the narrator; only `POST /v1/reports/{id}/generate` does.

## Model configuration

| Item | Value |
|---|---|
| Preference order | `qwen3:8b` → `llama3.2:3b` → `phi4-mini` (first locally available wins; configurable in `configs/ollama.yaml`) |
| Note | The prompt spec suggests `qwen3.5:4b`; that model is not published for Ollama on this machine — `qwen3:8b` is the installed nearest equivalent, and the substitution is recorded here. |
| Temperature | 0 |
| Format | Ollama JSON mode + schema-constrained prompt |
| Timeout / retries | 45 s / 1 retry |
| Circuit breaker | opens after 3 consecutive failures, 300 s cooldown |
| Network | localhost only; **no customer data leaves the machine** |

## Hallucination validator — rejection rules

Output is rejected (with machine-readable reasons) when it:

1. is invalid JSON or fails the schema (`NarratorOutput`),
2. changes `verified_risk_score` (must equal the input calibrated risk exactly),
3. changes `risk_tier`,
4. mentions any `Fxxxx` feature not present in the supplied facts,
5. invents a currency amount,
6. asserts criminal guilt (regex family: guilty/fraudster/launderer/…),
7. recommends an action outside the analyst-check allowlist,
8. omits the required limitations (intent + human review).

Every rule has a dedicated unit test that **forces** the malformed output and
asserts rejection (`tests/unit/test_llm_guardrails.py`, 13 tests). The demo
pipeline plants a hallucinated output (invented amount + guilt claim + tier
change + score change + disallowed action) and records its rejection in
`artifacts/evidence/demo_scenarios.json`.

## Outage behaviour

With Ollama stopped: `available_model()` returns None → the deterministic
template answers immediately. `GET /health/ready` reports
`"ollama_required": false`. Scoring latency and results are unchanged
(verified by integration tests running with no Ollama process).

## Measured events

Validator rejections and fallback events are logged and counted in the
narrative payload (`llm_rejected_reasons`) stored with each generated report,
so the hallucination-rejection rate is auditable from the database.
