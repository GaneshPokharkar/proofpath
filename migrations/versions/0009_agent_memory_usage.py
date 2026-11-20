"""Chat agent memory (anonymous sessions) and anonymous daily usage counts.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-20
"""

from alembic import op

from app.schema_live import STEP_0009

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for sql in STEP_0009:
        op.execute(sql)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_sessions")
    op.execute("DROP TABLE IF EXISTS usage_daily")
