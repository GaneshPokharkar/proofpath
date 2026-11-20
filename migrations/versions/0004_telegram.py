"""Telegram chats that talk to the bot, with their saved alert search.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-20
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE telegram_chats (
            chat_id       BIGINT PRIMARY KEY,
            username      TEXT,
            alert_query   TEXT,                      -- e.g. "python intern pune"
            alerts_on     BOOLEAN NOT NULL DEFAULT false,
            last_alert_at TIMESTAMPTZ,               -- only jobs first seen after this are sent
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_telegram_chats_alerts ON telegram_chats (alerts_on) WHERE alerts_on")


def downgrade() -> None:
    op.execute("DROP TABLE telegram_chats")
