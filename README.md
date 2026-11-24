# ProofPath

**Live demo: [proofpath-iota.vercel.app](https://proofpath-iota.vercel.app)** · Python · FastAPI ·
PostgreSQL + pgvector · LLM agent with tool calling · CI/CD on GitHub Actions · Vercel + Neon

A full-stack AI job-search app for students in India. Search live, de-duplicated job postings,
upload a resume to see a match score on every job and exactly what to change to reach 100%, ask an
assistant in plain words ("python internships in Pune, save the first one"), and get a tailored,
fact-checked one-page resume PDF for a specific job.

## What to look at (5 minutes)

| Area | Where | What it shows |
|---|---|---|
| Backend and API | `app/api/main.py`, `app/search.py` | REST API, hybrid search (full-text + vectors, reciprocal rank fusion) |
| Data pipeline | `app/ingest/` | Daily ingestion from Greenhouse, Lever, Ashby; cleaning, de-duplication, deadlines |
| LLM agent | `app/agent/` | Function calling over 6 tools, two-layer memory, hallucination guard |
| Resume engine | `app/resume/` | Structured extraction, per-job scoring, rewrite + verifier, LaTeX PDF |
| LLMOps | `app/llmops/` | Model router with fallback, per-call tracing, semantic cache |
| Quality | `tests/`, `evals/`, `.github/workflows/` | Automated tests on real Postgres, 100-case agent evals, CI/CD with smoke tests |
| Design notes | `docs/ARCHITECTURE.md`, `docs/learn/` | Architecture, pipeline, and the reasoning behind each design choice |

Run locally: `make setup db run` (Docker for Postgres), then `make check` runs what CI runs.

## Architecture: fetch once, serve many

ProofPath never searches job portals while a student is waiting. Job boards are read once a day,
stored in Postgres, and every request after that is answered from the database.

```
 once a day (Vercel Cron)                       every student request
 ────────────────────────                       ─────────────────────
 Greenhouse ─┐                                  /search ─► hybrid search over stored jobs
 Lever ──────┤   ingest: normalise, dedupe,     /market ─► saved skill demand for the role,
 Arbeitnow ──┼─► hash, embed changed jobs, ───►            or computed once and saved
 Remotive ───┘   classify field + level,        /resume ─► saved demand + one resume read
                 read last date to apply,       /readiness ─► saved demand + GitHub evidence
                 retire closed + expired jobs
                              │
                     PostgreSQL + pgvector (Neon)
       jobs · job_embeddings · job_skills · market_cache · ingestion_runs
```

| What is stored | Computed | Reused until |
| --- | --- | --- |
| Job postings, cleaned text, field and level | once per posting, at ingest | the posting changes (content hash) |
| Embeddings (title and whole posting) | once per posting version | the posting changes |
| Skills each posting asks for (LLM) | once per posting version | the posting changes |
| Skill demand per role (`market_cache`) | first time any student asks for that role | the next successful refresh |
| Last date to apply (`apply_by`) | at ingest, from a source field or the posting text | the date passes: hidden at once, retired at the next refresh |

So the second student who asks "what skills does a data analyst need?" gets the answer from the
database with no new LLM calls, and no student ever sees a job whose last date to apply has passed.
A daily refresh touches only new or changed postings (about 20 seconds after the first load).

**Last date to apply.** Few boards publish a deadline field, so ProofPath also reads phrases such as
"Application deadline: December 16, 2026" or "Last date to apply: 30/11/2026" from the posting text
(`app/ingest/deadline.py`). Only dates that follow such a phrase count, and they must fall after the
posting date. Jobs without a stated deadline show when the refresh last confirmed them open, and they
are retired as soon as they disappear from the company's board.

**Search.** Four ranked lists, fused with reciprocal rank fusion: full-text on the title, full-text on
the whole posting, vector similarity on the title, vector similarity on the whole posting. Title-only
lists stop long company boilerplate from drowning out what the job is. Filters (field, level, remote,
location, company, recency) apply to all four.

## Run it

**With Docker** (Postgres, migrations and API):

```bash
docker compose up --build -d
docker compose run --rm ingest          # fetch jobs from every source (first load ~20 min on 2 CPUs; later runs ~20 s)
curl "localhost:8000/search?q=python+backend+intern&remote=true"
```

**Without Docker** (needs Python 3.12+ and Postgres 16 with pgvector):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                    # then edit the database URLs
alembic upgrade head
python -m app.cli ingest                # or: --only greenhouse:groww lever
python -m app.cli search "devops kubernetes" --location india
uvicorn app.api.main:app --reload
```

### API

| Endpoint | What it returns |
| --- | --- |
| `GET /search?q=...&remote=&location=&company=&posted_within_days=&source=&limit=` | ranked jobs, each with its keyword rank and semantic rank, plus query time |
| `GET /jobs/{id}` | full posting |
| `GET /stats` | active jobs per source and each source's last ingestion run |
| `GET /health` | liveness check (used by the Docker healthcheck) |

## Tests

The suite runs against a real PostgreSQL + pgvector database (no mocks for SQL), migrated with Alembic.
Parsers are tested on recorded API responses; the pipeline and search use a deterministic test embedder.

```bash
export JOBPILOT_TEST_DATABASE_URL=postgresql+psycopg://...   # an empty database
pytest -q
```

CI (GitHub Actions) runs lint, format check and the tests against a `pgvector/pgvector:pg16` service
container, then builds the Docker image.

## Data sources and terms

Only public APIs that publishers provide for this purpose are used; nothing is scraped. Every result keeps
its original URL and its source. Remotive asks API users to link back, credit Remotive, fetch at most four
times a day, and not use its jobs to collect signups, so Remotive jobs are excluded from alert emails.

## Roadmap

1. ~~Data and search: schema, ingestion from 4 sources, local embeddings, hybrid search, API, tests, CI~~
2. Chat agent: Claude tool use (6 tools), validated tool arguments, conversation summary plus long-term
   profile memory, grounding check, web chat UI
3. Automations: daily job alerts (Telegram/email), application tracker with follow-up drafts, resume
   tailoring to a one-page LaTeX PDF with a no-invented-claims verifier
4. Evals: 100-case golden set (tool choice, filter extraction, hallucinated-job rate), gated in CI
5. DevOps: Terraform on AWS (EC2, RDS Postgres with pgvector, ECR, SSM), CI/CD deploys, Prometheus and Grafana

## Privacy: nothing is kept longer than a day

- Resumes are read in memory and never written anywhere; contact details are removed before the AI sees them.
- Visitors' Gemini keys stay in their own browser and travel only with their own requests.
- The daily cron runs `app/retention.py`, which deletes everything older than 24 hours: GitHub checks, saved skill reports, Telegram subscriptions, old refresh logs, and job postings no longer seen. Postings are only cleared after a successful refresh, so a failed scrape never empties the site.

## From development to production

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): local setup (`make setup db run`), the CI
checks every pull request must pass (`make check` runs the same locally), Vercel preview and
production deploys, post-deploy smoke tests, migrations, nightly checks and rollback.

## Chat agent (Ask ProofPath)

`POST /chat` runs one turn of an LLM agent (Gemini function calling) over six tools in
`app/agent/tools.py`: `search_jobs`, `get_job`, `compare_jobs`, `shortlist`, `save_preference`,
`explain_match`. Arguments are validated; a bad call returns the error to the model so it can fix
it. Replies may cite jobs only as `[job:ID]` for ids a tool returned, and any other id is stripped
before the student sees it (and counted as a hallucination).

**Two-layer memory** (`app/agent/memory.py`), in PostgreSQL under an anonymous browser-generated id:
the last 6 turns verbatim plus a rolling LLM summary of older turns, and a long-term profile (skills,
locations, remote, levels, fields, salary floor, rejected roles, shortlist) written only through the
tools and injected into every turn. Rows are deleted after `MEMORY_DAYS` (default 1) without use,
or at once with `DELETE /chat/{id}` ("forget me").

## Evals

`evals/cases.jsonl` holds 100 fixed cases (search filters, preferences, shortlist, compare, explain,
details, small talk) run against `evals/jobs.json`, a snapshot of 173 real postings.
`python -m evals.run` reports tool-choice accuracy, filter-extraction accuracy and hallucinated-job
rate, and exits non-zero below `evals/thresholds.json`. `.github/workflows/evals.yml` runs it on
every change to the agent, prompts, model settings or evals (add a `GEMINI_API_KEY` repo secret).

First full run (gemini-flash-lite-latest, 2026-11-20, `evals/baseline.json`): tool-choice accuracy
0.99, filter-extraction accuracy 0.94, hallucinated-job rate 0.0 (no invented job ids), 1 timeout.

## Resume tailoring (ResumeForge)

`POST /resume/tailor` parses the resume into validated JSON, rewrites bullets for one posting,
and runs a verifier on every rewrite: a rule check (no new numbers outside `[brackets]`, no job
skill the resume never shows) and then an LLM fact-check against the original. Rejected lines keep
the student's wording. The result is rendered into a LaTeX template, compiled to PDF
(`LATEX_COMPILE_URL`), and trimmed bullet by bullet until it fits one page. Contact details are
taken out with regexes and never sent to the LLM.

Sources now include Greenhouse, Lever, **Ashby**, Arbeitnow and Remotive.
