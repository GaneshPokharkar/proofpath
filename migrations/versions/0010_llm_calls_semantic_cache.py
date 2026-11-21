"""LLM call traces and the semantic cache columns on market_cache.

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-21
"""

from alembic import op

from app.schema_live import STEP_0010

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for sql in STEP_0010:
        op.execute(sql)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS llm_calls")
    op.execute("ALTER TABLE market_cache DROP COLUMN IF EXISTS role_vec")
    op.execute("ALTER TABLE market_cache DROP COLUMN IF EXISTS scope")
