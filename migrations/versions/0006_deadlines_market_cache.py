"""Application deadlines on jobs, and a cache of market reads per role.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-20
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Last date to apply, when the source states one (field or a phrase in the description).
    op.execute("ALTER TABLE jobs ADD COLUMN apply_by DATE")
    op.execute("CREATE INDEX ix_jobs_apply_by ON jobs (apply_by) WHERE apply_by IS NOT NULL")
    op.execute(
        """
        -- Skill demand per role, computed once and served to every later student asking the same thing.
        -- A row is valid until the next successful job refresh (built_at older than it = stale).
        CREATE TABLE market_cache (
            key        TEXT PRIMARY KEY,           -- normalised role + filters + sample size
            role       TEXT NOT NULL,
            payload    JSONB NOT NULL,             -- serialised MarketRead
            built_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            hits       INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE market_cache")
    op.execute("ALTER TABLE jobs DROP COLUMN apply_by")
