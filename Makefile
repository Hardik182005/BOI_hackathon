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

addendum: train-select-v2 build-lenses-v2 shield robustness merchant-verifier audit-labels

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
