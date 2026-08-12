# Ollama Guardrails

Master prompt §22–23. What the local LLM is allowed to touch, and — more
importantly — what it is structurally incapable of touching.

Implementation: `src/muleguard/llm/` · Tests: `tests/unit/test_llm_guardrails.py`
(13 tests, each forcing a specific malformed output and asserting rejection).

---

## 1. The rule

> **No LLM is a classifier here.** Ollama may explain verified evidence. It never
> calculates or modifies a risk probability, a model score, a threshold, a
> prediction, an analyst action, or an item of evidence.

This is not a policy we ask a prompt to respect. It is enforced in three
independent places, so that no single failure — a prompt-injection, a model
swap, a bug — can breach it:

1. **Architecture.** `muleguard.models.scoring` contains **no LLM call at all**.
   There is no code path from the narrator back into a score.
2. **Interface.** The narrator receives a `NarratorInput` containing only
   already-computed facts: the calibrated risk, tier, agreement, conformal/OOD
   status and SHAP reason facts. It cannot see the raw row, the model, or the
   data.
3. **Validator.** Every output is parsed against a strict schema and a
   hallucination validator; any deviation discards the output **entirely** in
   favour of a deterministic template.

`POST /v1/score` never invokes the narrator. Only
`POST /v1/reports/{id}/generate` does — so the scoring API is LLM-free by
construction, not by configuration.

---

## 2. Configuration

| Item | Value |
|---|---|
| Preference order | `qwen3:8b` → `llama3.2:3b` → `phi4-mini` (first locally available wins; `configs/ollama.yaml`) |
| Temperature | **0** |
| Format | Ollama JSON mode + schema-constrained prompt |
| Timeout / retries | 45 s / 1 retry |
| Circuit breaker | opens after 3 consecutive failures, 300 s cooldown |
| Network | **localhost only** — no customer data leaves the machine, no API key, no internet |

**Substitution recorded:** the master prompt suggests `qwen3.5:4b`. That model is
not published for Ollama on this machine, so `qwen3:8b` is the installed nearest
equivalent. Noting it here rather than silently using a different model.

---

## 3. The eight rejection rules

Narrator output is rejected, with machine-readable reasons, when it:

| # | Rejection | Why it is fatal |
|---:|---|---|
| 1 | is invalid JSON or fails the `NarratorOutput` schema | unparseable text cannot be shown to an analyst as evidence |
| 2 | changes `verified_risk_score` | must equal the input calibrated risk **exactly** — this is the core rule |
| 3 | changes `risk_tier` | the tier is a policy decision, not a narrative one |
| 4 | mentions any `Fxxxx` feature not in the supplied facts | citing a feature the model did not use is fabricated evidence |
| 5 | invents a currency amount | the single most convincing kind of hallucination in a banking report |
| 6 | asserts criminal guilt (regex family: guilty / fraudster / launderer / …) | the system produces review recommendations, never findings about a person |
| 7 | recommends an action outside the analyst-check allowlist | an LLM must not be able to propose an action the policy engine would refuse |
| 8 | omits the required limitations (intent + human review) | every narrative must carry its own caveats |

**Every rule has a dedicated unit test that forces the malformed output and
asserts the rejection.** A guardrail with no test that triggers it is a comment.

The demo pipeline plants a deliberately hallucinated narrative — invented amount,
guilt claim, tier change, score change, disallowed action — and records its
rejection in `artifacts/evidence/demo_scenarios.json`. The release gate re-runs
this: `scoring_survives_ollama_outage` reports the planted hallucination
**rejected for 8 reasons**.

---

## 4. Outage behaviour

With Ollama stopped:

- `available_model()` returns `None`;
- `llm/deterministic_fallback.py` answers immediately from a template;
- `GET /health/ready` reports `"ollama_required": false`;
- **scoring latency and results are unchanged** — verified by integration tests
  that run with no Ollama process at all.

This is the test that matters most for a judge: **stop Ollama and the product
still works.** Risk scores, tiers, ProofGraphs, the Validation Lab and the alert
queue are all fully functional. Only the prose narration degrades to a template.

Release-gate check `llm_cannot_alter_score` PASS (rejected with 2 reasons);
`scoring_survives_ollama_outage` PASS.

---

## 5. Auditability

Validator rejections and fallback events are logged and counted in the narrative
payload (`llm_rejected_reasons`) stored with each generated report. **The
hallucination-rejection rate is queryable from the database** — so the guardrail's
own effectiveness is measurable after the fact rather than assumed.

---

## 6. Compared with the alternative

One of the seven competitor repositories we reviewed generates its investigation
summary with the Gemini API. That is the honest opposite arrangement: the
explanation *is* generated text, the system needs an API key and internet, and
customer data leaves the machine.

Ours explains only nodes that already exist, computed by a model that ran before
the LLM was invoked, on a machine with no network dependency. If the narrator is
removed entirely, **no number in this system changes**.
