"""Remember when a job was last shown in someone's search, so only searched jobs are kept.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-20
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN last_served_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN last_served_at")
