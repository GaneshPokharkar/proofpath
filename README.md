# JobPilot

An AI job-search agent. You describe the job you want in plain language; JobPilot searches a live,
daily-refreshed index of real job postings and explains why each one fits.

> **Status:** phase 1 of 5 is done: the jobs index and hybrid search. The Claude-powered chat agent,
> automations (daily alerts, application tracker, resume tailoring), evals and the AWS deployment
> follow. See [Roadmap](#roadmap).

## How it works (phase 1)

```
 Public job APIs                    PostgreSQL 16 + pgvector
 ───────────────                    ─────────────────────────────────────────
 Remotive ─┐                        jobs            (tsvector column, GIN index)
 Arbeitnow ├─► ingest pipeline ───► job_embeddings  (384-d vectors, HNSW index)
 Greenhouse│   normalise, dedupe,   ingestion_runs  (audit trail per source)
 Lever ────┘   hash, embed changed            │
                                              ▼
                     FastAPI  /search ◄── hybrid search (full-text + vector, RRF)
```

**Ingestion** (`app/ingest/`) pulls from four public job APIs, configured in [`sources.yaml`](sources.yaml):
Remotive and Arbeitnow, plus company boards on Greenhouse (Groww, InMobi, Databricks, GitLab, Anthropic, ...)
and Lever (CRED, Meesho, Zeta, Paytm, ...). Each posting is normalised to one shape, HTML is converted to
text, and a SHA-256 content hash decides what changed, so only new or edited postings are re-embedded.
Closed jobs are retired: immediately for full-snapshot boards, after 14 days unseen for partial feeds.
Every source runs in its own transaction and writes an `ingestion_runs` row, so a broken board is recorded
and reported without blocking the rest. An empty response from a board that had jobs is treated as an
error rather than a reason to retire everything.

**Embeddings** run locally with [fastembed](https://github.com/qdrant/fastembed) (BAAI/bge-small-en-v1.5,
ONNX, CPU): no API key and no per-call cost.

**Hybrid search** (`app/search.py`) runs two retrievers inside one SQL query and fuses them:

| Retriever | Good at | Weak at |
| --- | --- | --- |
| Postgres full-text (`ts_rank_cd`, title weighted highest) | exact terms: "Kubernetes", "FastAPI", company names | paraphrases |
| pgvector cosine similarity (HNSW) | meaning: "backend intern" ≈ "software engineering internship, APIs" | exact rare terms |

The two ranked lists are merged with **reciprocal rank fusion**, `score = Σ 1 / (60 + rank)`. RRF uses only
ranks, so a text-rank score and a cosine distance never have to be put on the same scale. Chat-style queries
are OR-ed rather than AND-ed, so "python backend intern remote" still matches jobs that contain most of
those words. Filters (remote, location, company, recency, source) apply to both retrievers.

## Run it

**With Docker** (Postgres, migrations and API):

```bash
docker compose up --build -d
docker compose run --rm ingest          # fetch jobs from every source (first load takes ~10 min)
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
