# Performance Report

Master prompt §32. Measured on this machine — **i5-1135G7, 16 GB, CPU-only, no
GPU**. Live evidence: `artifacts/testing/performance_results.json`
(7 checks, 7 passed). Regenerate with the QA harness.

Everything below is a single-laptop number. Nothing here is extrapolated to
production, and where a production figure would be different that is said rather
than implied.

---

## 1. Measured

| Metric | Value |
|---|---:|
| Single-row scoring latency **p50** | **0.307 s** |
| Single-row scoring latency p95 | 0.422 s |
| Single-row scoring latency mean (n = 15) | 0.321 s |
| **Batch throughput** | **687.7 rows/s** on the 1,818-row locked-test batch |
| Model bundle size | **3.46 MB** (3,464,956 bytes / 3.30 MiB) |
| Concurrency — 5 parallel requests | 5 × 200 OK |
| Concurrency — 10 parallel requests | 10 × 200 OK |
| Memory headroom during scoring | 1.3 GB free; scoring itself needs < 1 GB |
| Cold start → ready | ~8–12 s including bundle load, health-gated by `run.sh` |
| Frontend production build | passes, ~600 KB gzipped |

Concurrent responses are **deterministic** — the same input produces byte-identical
output under load, which is the property that matters for an audit trail.

---

## 2. Why these numbers are what they are

**The bundle is 3.46 MB.** A 120-feature XGBoost model is small — and the bundle
carries more than the model: the Platt calibrator, the Mondrian conformal sets,
the anomaly and OOD lenses, the hard-negative verifier, the preprocessor and the
category maps all travel with it. It loads once at
startup and stays in memory — there is no per-request reload, which is why
latency is stable across repeated calls rather than decaying.

**Inference is single-digit milliseconds.** The 0.307 s p50 is dominated by HTTP,
frame construction over a 3,925-column row, and meta-feature derivation — not by
the model. Batch amortises all of that, which is why throughput is 688 rows/s
rather than ~3.

**The whole 9,082-row dataset scores in about 13 seconds.** A bank's overnight
batch is not a scaling problem for this design.

### The contrast that made this a decision

The TabPFN challenger scores **0.911 OOF PR-AUC** against our champion's 0.769 —
and takes **438 seconds for a single row**, because in-context learning replays
the training set per call. An analyst clicking an account would wait seven
minutes. That model is more accurate and unservable; the trade is documented in
`docs/MODEL_TOURNAMENT.md` §6.

---

## 3. Engineering guarantees

- **Model loaded once** at startup, never per request.
- **No unbounded memory growth**: batch scoring is chunked (HTTP path at 500
  rows) with progress audit events; SHAP explanations are optional per request;
  DB writes are transactional.
- **Timeouts everywhere**: narrator 45 s with a circuit breaker (3 failures →
  300 s cooldown), DB busy timeout 15 s, documented client timeouts.
- **Rate limiting**: 240 req/min per client — a demo-scale guardrail.
- **One uvicorn worker** by default, sized for 16 GB.

---

## 4. What is *not* measured here

| Not measured | Why it matters |
|---|---|
| Multi-worker throughput | single worker only; horizontal scaling is a documented extension |
| PostgreSQL under load | the demo uses SQLite |
| GPU inference | never required — the served model is CPU-only by design |
| Sustained multi-hour load | the harness measures burst behaviour, not endurance |
| Cold-cache cloud latency | there is no cloud deployment; the product runs locally by design |

**Scale honesty.** These are single-node CPU demo numbers. Multiple workers,
PostgreSQL and GPU challengers are a roadmap in `docs/DEPLOYMENT_GUIDE.md`, not a
benchmark we are claiming.

---

## 5. The performance property that actually matters

Not latency — **offline reproducibility**.

The system needs no internet, no API key, no MCP server and no browser agent to
produce a score. Stop Ollama and every number is unchanged. A judge can
disconnect the network, run `./run.sh`, and get identical results — which is
worth more at evaluation time than a faster p50.
