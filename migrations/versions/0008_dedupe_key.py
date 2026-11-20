"""One copy of each job: the same company + title + location from two boards is one job.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-20
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

KEY = "md5(lower(regexp_replace(company, '[^a-zA-Z0-9]', '', 'g')) || '|' || lower(regexp_replace(title, '[^a-zA-Z0-9]', '', 'g')) || '|' || lower(regexp_replace(coalesce(location, ''), '[^a-zA-Z0-9]', '', 'g')))"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS dedupe_key TEXT GENERATED ALWAYS AS ({KEY}) STORED"
    )
    op.execute("CREATE INDEX IF NOT EXISTS jobs_dedupe_key ON jobs (dedupe_key)")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN dedupe_key")
