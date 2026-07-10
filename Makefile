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

evaluate-locked-test:
	$(PY) -m muleguard.cli.evaluate

plots:
	$(PY) -m muleguard.evaluation.plots

test:
	$(PY) -m pytest

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
