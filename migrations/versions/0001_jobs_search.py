"""Jobs index: jobs (full-text columns), job_embeddings (pgvector HNSW), ingestion_runs.

Revision ID: 0001
Revises:
Create Date: 2026-11-20
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # array_to_string() is only STABLE, so it cannot appear in a generated column directly.
    # Joining with a space is deterministic, so this wrapper is safely IMMUTABLE.
    op.execute(
        "CREATE FUNCTION jobpilot_tags_text(text[]) RETURNS text "
        "LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT array_to_string($1, ' ') $$"
    )

    op.execute(
        """
        CREATE TABLE jobs (
            id              BIGSERIAL PRIMARY KEY,
            source          VARCHAR(64)  NOT NULL,
            external_id     TEXT NOT NULL,
            title           TEXT NOT NULL,
            company         TEXT NOT NULL,
            location        TEXT,
            remote          BOOLEAN,
            employment_type TEXT,
            tags            TEXT[] NOT NULL DEFAULT '{}',
            salary_text     TEXT,
            description     TEXT NOT NULL,
            url             TEXT NOT NULL,
            posted_at       TIMESTAMPTZ,
            first_seen_at   TIMESTAMPTZ NOT NULL,
            last_seen_at    TIMESTAMPTZ NOT NULL,
            content_hash    VARCHAR(64) NOT NULL,
            is_active       BOOLEAN NOT NULL DEFAULT true,
            -- Title weighs most, then company and tags, then the description body.
            search_tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(company, '')), 'B') ||
                setweight(to_tsvector('english', jobpilot_tags_text(tags)), 'B') ||
                setweight(to_tsvector('english', coalesce(description, '')), 'C')
            ) STORED,
            -- Title on its own: searched as a separate list so a title match is never drowned out
            -- by long descriptions that open with company boilerplate.
            title_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, ''))) STORED,
            CONSTRAINT uq_jobs_source_external_id UNIQUE (source, external_id)
        )
        """
    )
    op.execute("CREATE INDEX ix_jobs_search_tsv ON jobs USING GIN (search_tsv)")
    op.execute("CREATE INDEX ix_jobs_title_tsv ON jobs USING GIN (title_tsv)")
    # Partial index: almost every query filters to active jobs.
    op.execute("CREATE INDEX ix_jobs_active_posted ON jobs (posted_at DESC) WHERE is_active")
    op.execute("CREATE INDEX ix_jobs_source_last_seen ON jobs (source, last_seen_at)")

    op.execute(
        """
        CREATE TABLE job_embeddings (
            job_id       BIGINT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
            model        TEXT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            embedding    vector(384) NOT NULL,   -- title + company + location + tags + description
            title_embedding vector(384) NOT NULL, -- title alone
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_job_embeddings_hnsw ON job_embeddings "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute(
        "CREATE INDEX ix_job_embeddings_title_hnsw ON job_embeddings "
        "USING hnsw (title_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.execute(
        """
        CREATE TABLE ingestion_runs (
            id           BIGSERIAL PRIMARY KEY,
            source       VARCHAR(64) NOT NULL,
            started_at   TIMESTAMPTZ NOT NULL,
            finished_at  TIMESTAMPTZ,
            status       VARCHAR(16) NOT NULL,
            fetched      INTEGER NOT NULL DEFAULT 0,
            inserted     INTEGER NOT NULL DEFAULT 0,
            updated      INTEGER NOT NULL DEFAULT 0,
            unchanged    INTEGER NOT NULL DEFAULT 0,
            deactivated  INTEGER NOT NULL DEFAULT 0,
            embedded     INTEGER NOT NULL DEFAULT 0,
            error        TEXT,
            CONSTRAINT ck_ingestion_runs_status CHECK (status IN ('running', 'success', 'failed'))
        )
        """
    )
    op.execute("CREATE INDEX ix_ingestion_runs_source_started ON ingestion_runs (source, started_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE ingestion_runs")
    op.execute("DROP TABLE job_embeddings")
    op.execute("DROP TABLE jobs")
    op.execute("DROP FUNCTION jobpilot_tags_text(text[])")
