# Agent evals

- `cases.jsonl`: 100 fixed cases (made by `make_cases.py`).
- `jobs.json`: 173 real postings the cases run against (made by `build_fixture.py`).
- `run.py`: runs every case with the real model; fails below `thresholds.json`.
- `baseline.json`: the first full run.

Run locally: `python -m evals.run` (needs `JOBPILOT_DATABASE_URL` and `JOBPILOT_LLM_API_KEY`).
In CI: `.github/workflows/evals.yml`, using the `GEMINI_API_KEY` repository secret.
