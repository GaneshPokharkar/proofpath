"""Job field (software, data & AI, ...) and level (intern, entry, ...), classified from titles.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-20
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN field VARCHAR(16) NOT NULL DEFAULT 'other'")
    op.execute("ALTER TABLE jobs ADD COLUMN level VARCHAR(8) NOT NULL DEFAULT 'mid'")
    op.execute("CREATE INDEX ix_jobs_field_level ON jobs (field, level) WHERE is_active")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN level")
    op.execute("ALTER TABLE jobs DROP COLUMN field")
