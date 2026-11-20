# Day-to-day commands. Local Postgres comes from docker compose.
TEST_DB ?= postgresql+psycopg://jobpilot:jobpilot@localhost:5432/jobpilot_test
DB ?= postgresql+psycopg://jobpilot:jobpilot@localhost:5432/jobpilot

.PHONY: setup db run test lint security check evals smoke

setup:            ## install dev tools and the git hook
	pip install -r requirements-dev.txt pre-commit && pre-commit install

db:               ## start Postgres + pgvector and apply migrations
	docker compose up -d db
	docker compose exec -T db sh -c 'until pg_isready -U jobpilot; do sleep 1; done; psql -U jobpilot -tc "SELECT 1 FROM pg_database WHERE datname='"'"'jobpilot_test'"'"'" | grep -q 1 || psql -U jobpilot -c "CREATE DATABASE jobpilot_test"'
	JOBPILOT_DATABASE_URL=$(DB) alembic upgrade head

run:              ## the app on http://localhost:8000 with reload
	JOBPILOT_DATABASE_URL=$(DB) uvicorn app.api.main:app --reload

lint:
	ruff check . && ruff format --check .

security:
	pip-audit -r requirements.txt && bandit -q -c pyproject.toml -r app -ll

test:
	JOBPILOT_TEST_DATABASE_URL=$(TEST_DB) pytest -q --cov=app --cov-fail-under=80

check: lint security test   ## everything CI runs, before you push

evals:            ## the 100 agent cases against a real model (needs JOBPILOT_LLM_API_KEY)
	JOBPILOT_DATABASE_URL=$(DB) python -m evals.run

smoke:            ## smoke-test a deployment: make smoke URL=https://...
	python scripts/smoke.py $(URL)
