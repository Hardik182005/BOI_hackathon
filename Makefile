# MuleGuard - Trinetra. Windows: use `make` from Git Bash, or copy commands.
PY := .venv/Scripts/python.exe

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .[dev]

audit-env:
	$(PY) -m muleguard.cli.audit_env

audit-data:
	$(PY) -m muleguard.cli.audit_data

make-splits:
	$(PY) -m muleguard.cli.make_splits

train-baselines:
	$(PY) -m muleguard.cli.train baselines

train-select:
	$(PY) -m muleguard.cli.tournament select

train-tune:
	$(PY) -m muleguard.cli.tournament tune

train-core:
	$(PY) -m muleguard.cli.tournament finalists

train-advanced:
	$(PY) -m muleguard.cli.advanced

build-lenses:
	$(PY) -m muleguard.cli.build_lenses

# --- addendum (competitor-driven upgrades) ---------------------------------
# Run in this order: the tournament reopens champion selection, the shield and
# robustness suites judge the champion it picks, and the verifier and label
# audit are read-only observations layered on top. None of them retrains
# against validation data.
train-select-v2:
	$(PY) -m muleguard.cli.tournament_v2

build-lenses-v2:
	$(PY) -m muleguard.cli.build_lenses_v2

shield:
	$(PY) -m muleguard.cli.shield_v2

robustness:
	$(PY) -m muleguard.cli.robustness_v2

merchant-verifier:
	$(PY) -m muleguard.cli.merchant_verifier

audit-labels:
	$(PY) -m muleguard.cli.audit_labels

# Re-reads the stored dev OOF predictions of the promoted champion. Nothing is
# refitted, so this is safe to run any time the champion or the policy changes.
capacity-curve:
	$(PY) -m muleguard.cli.capacity_curve

addendum: train-select-v2 build-lenses-v2 shield robustness merchant-verifier audit-labels capacity-curve

# Section 42. Needs a backend on :8001 - it goes over HTTP on purpose, because
# the organiser will too.
dry-run:
	$(PY) -m muleguard.cli.dry_run

evaluate-locked-test:
	$(PY) -m muleguard.cli.evaluate

plots:
	$(PY) -m muleguard.evaluation.plots

run:
	bash run.sh

stop:
	bash scripts/stop.sh

test:
	$(PY) -m pytest

test-ml:
	bash scripts/test_ml.sh

test-backend:
	bash scripts/test_backend.sh

test-frontend:
	bash scripts/test_frontend.sh

test-e2e:
	bash scripts/test_e2e.sh

test-security:
	bash scripts/test_security.sh

test-offline:
	bash scripts/test_offline.sh

release-test:
	bash scripts/release_test.sh

serve-api:
	$(PY) -m uvicorn muleguard.api.main:app --host 127.0.0.1 --port 8001

serve-frontend:
	cd frontend && npm run dev

demo:
	$(PY) -m muleguard.cli.demo

docker-up:
	docker compose up --build

export-submission:
	$(PY) -m muleguard.cli.export_submission

release-gate:
	$(PY) -m muleguard.cli.release_gate

# Section 56. The sixteen required figures, from saved predictions only. A plot
# whose evidence does not exist yet is reported as skipped, never faked.
plots-final:
	$(PY) -m muleguard.evaluation.plots_final

# Global mean |SHAP| for the importance figure. Refits each fold to attribute
# out-of-fold, so it costs a few minutes; not part of the demo path.
global-shap:
	$(PY) -m muleguard.cli.global_shap

# Sections 57-58. Regenerates every spec-named artifact from its real source and
# rewrites docs/ARTIFACT_AND_REPORT_INDEX.md. Safe to run any time.
reconcile-artifacts:
	$(PY) -m muleguard.cli.reconcile_artifacts

# Section 60. Fails when an artifact under artifacts/metrics/ has no ledger
# entry - that is the "no forgotten experiments" rule, enforced not remembered.
experiment-ledger:
	$(PY) -m muleguard.cli.experiment_ledger

# Does the shipped champion survive the nested (primary) protocol? Exit 0
# confirmed, 1 challenged, 2 while the nested run is incomplete. Read-only.
nested-promotion:
	$(PY) -m muleguard.cli.nested_promotion

# Resolves docs/TUNING_OVERFIT_HYPOTHESIS.md by pairing the tuned and untuned
# arms on all 15 outer folds. Reads stored predictions; retrains nothing.
tuning-overfit:
	$(PY) -m muleguard.cli.tuning_overfit

# Section 58. The nested tournament write-up, generated from the nested run's
# own artifacts - including the verdict on the shipped champion.
nested-report:
	$(PY) -m muleguard.cli.nested_report

# Section 58. The top-level validation report: the argument around the A-L
# answer, generated from artifacts so it cannot drift. Always exits 0 - it
# reports, it does not decide.
final-report:
	$(PY) -m muleguard.cli.final_report

# Section 65. Assembles the A-L final response from artifacts. Exit 0 only when
# every blocker is clear and every criterion met; 2 while evidence is open.
final-verdict:
	$(PY) -m muleguard.cli.final_verdict

# Section 59. Default mode never retrains - that is the judge demo path.
final-validation:
	bash scripts/final_validation.sh

final-validation-retrain:
	bash scripts/final_validation.sh --full-retrain
