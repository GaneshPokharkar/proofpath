# ProofPath

An AI job-search agent. You describe the job you want in plain language; JobPilot searches a live,
daily-refreshed index of real job postings and explains why each one fits.

> **Live:** student web app (find jobs, skills a role needs, resume check with project ideas,
> GitHub readiness), daily refresh, and a Telegram bot, on Vercel + Neon free tiers.

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
