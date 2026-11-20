"""Students, their GitHub repos, and skill evidence found in those repos.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-20
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE students (
            id           BIGSERIAL PRIMARY KEY,
            github_login TEXT NOT NULL UNIQUE,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE repos (
            id          BIGSERIAL PRIMARY KEY,
            student_id  BIGINT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            full_name   TEXT NOT NULL UNIQUE,         -- "owner/name"
            description TEXT,
            url         TEXT NOT NULL,
            is_fork     BOOLEAN NOT NULL,
            stars       INTEGER NOT NULL DEFAULT 0,
            pushed_at   TIMESTAMPTZ,
            languages   JSONB NOT NULL DEFAULT '{}',  -- {"Python": bytes, ...} from the GitHub API
            analyzed_at TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE INDEX ix_repos_student ON repos (student_id)")
    op.execute(
        """
        CREATE TABLE repo_evidence (
            repo_id  BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            skill_id BIGINT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            -- how the skill was detected: a language GitHub measured, a declared dependency,
            -- a file that only exists if the tool is used (Dockerfile, CI workflow), or the README
            source   VARCHAR(16) NOT NULL,
            detail   TEXT NOT NULL,                 -- e.g. "requirements.txt: fastapi==0.115"
            PRIMARY KEY (repo_id, skill_id, source),
            CONSTRAINT ck_repo_evidence_source CHECK (source IN ('language', 'dependency', 'file', 'readme'))
        )
        """
    )
    op.execute("CREATE INDEX ix_repo_evidence_skill ON repo_evidence (skill_id)")


def downgrade() -> None:
    op.execute("DROP TABLE repo_evidence")
    op.execute("DROP TABLE repos")
    op.execute("DROP TABLE students")
