# MuleGuard · Trinetra

**AI/ML classification of suspicious mule accounts — PS2, PSB Cybersecurity, Fraud & AI Hackathon 2026 (Bank of India × IIT Hyderabad). Team Kryptonite.**

> Sees the mule. Spares the look-alike. Never certifies the unseen.

MuleGuard scores bank accounts for mule-like behaviour with a calibrated, leakage-audited
tabular ML ensemble (Trinetra Lens 1), protects legitimate look-alikes via hard-negative
verification + conformal abstention (Lens 2), challenges misses with anomaly/OOD detection
(Lens 3), and routes every decision through a deterministic human-in-the-loop policy engine.
A local LLM (Ollama) may *narrate* verified results; it can never compute or change a score.

## Quick start

```bash
# 1. Environment (Windows; venv lives on E: by project convention)
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]

# 2. Data pipeline (raw file DataSet.xlsx at repo root; never modified)
.venv\Scripts\python -m muleguard.cli.audit_env       # environment snapshot
.venv\Scripts\python -m muleguard.cli.audit_data      # ingest + fingerprint + audit + quarantine
.venv\Scripts\python -m muleguard.cli.make_splits     # locked test + repeated CV folds
.venv\Scripts\python -m muleguard.cli.train baselines # dummy / logistic / LightGBM
.venv\Scripts\python -m muleguard.cli.train tournament
.venv\Scripts\python -m muleguard.cli.evaluate locked-test   # single touch, end of project

# 3. Tests
.venv\Scripts\python -m pytest

# 4. Serve
.venv\Scripts\python -m muleguard.cli.serve           # FastAPI on :8000
cd frontend && npm install && npm run dev             # dashboard on :5173
```

See `docs/` for the full report set (data audit, leakage audit, tournament, calibration,
lens report, model card, judge Q&A, release gate) and `Makefile` for shortcuts.

## Non-negotiables implemented

- `F3924` (target) and `F3912` (measured target leak) are quarantined from every model.
- Locked 20% stratified test set — created before training, touched exactly once.
- Primary metric: PR-AUC with bootstrap CIs + recall/precision at analyst alert budgets.
- Natural class prevalence preserved; no SMOTE in the accepted model.
- No automatic account freezing anywhere; analyst approval gates every high-impact action.
- Ollama optional: scoring is deterministic and fully functional with the LLM stopped.
