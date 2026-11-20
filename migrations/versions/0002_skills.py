"""Skills: canonical skills (with embeddings), aliases, skills extracted per job.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-20
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE skills (
            id         BIGSERIAL PRIMARY KEY,
            name       TEXT NOT NULL UNIQUE,          -- canonical display name, e.g. "PostgreSQL"
            category   VARCHAR(32) NOT NULL,
            embedding  vector(384) NOT NULL,          -- used to merge new spellings into existing skills
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT ck_skills_category CHECK (category IN
                ('language', 'framework', 'database', 'cloud', 'ml', 'devops', 'tool', 'concept'))
        )
        """
    )
    op.execute(
        """
        CREATE TABLE skill_aliases (
            alias    TEXT PRIMARY KEY,                -- lower-cased spelling as seen, e.g. "postgres"
            skill_id BIGINT NOT NULL REFERENCES skills(id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """
        CREATE TABLE job_skills (
            job_id     BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            skill_id   BIGINT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            importance VARCHAR(16) NOT NULL,
            PRIMARY KEY (job_id, skill_id),
            CONSTRAINT ck_job_skills_importance CHECK (importance IN ('required', 'preferred'))
        )
        """
    )
    op.execute("CREATE INDEX ix_job_skills_skill ON job_skills (skill_id)")
    op.execute(
        """
        -- One row per job whose skills have been extracted; content_hash makes the cache exact:
        -- an edited posting is extracted again, an unchanged one never is.
        CREATE TABLE job_skill_extractions (
            job_id       BIGINT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
            content_hash VARCHAR(64) NOT NULL,
            model        TEXT NOT NULL,
            extracted_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE job_skill_extractions")
    op.execute("DROP TABLE job_skills")
    op.execute("DROP TABLE skill_aliases")
    op.execute("DROP TABLE skills")
