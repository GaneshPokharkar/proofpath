"""Retention: nothing is kept per person; the shared job database is kept whole.

Every job ever scraped stays in one de-duplicated table (closed and expired ones are only hidden
from search), so later visitors reuse it. What goes after a day is anything tied to a person:
GitHub checks and Telegram chats without alerts. Resumes and API keys are never stored at all.
Shared skill reports per role are reused by everyone for 7 days.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

RETENTION = timedelta(days=1)
REPORT_KEEP = timedelta(days=7)


def purge(session: Session, now: datetime | None = None) -> dict[str, int]:
    cutoff = (now or datetime.now(UTC)) - RETENTION
    p = {"cutoff": cutoff, "report_cutoff": (now or datetime.now(UTC)) - REPORT_KEEP}
    done: dict[str, int] = {}

    def run(name: str, sql: str) -> None:
        done[name] = session.execute(text(sql), p).rowcount or 0

    run("skill_reports", "DELETE FROM market_cache WHERE built_at < :report_cutoff")
    run("github_checks", "DELETE FROM students WHERE created_at < :cutoff")
    from app.config import get_settings

    p["memory_cutoff"] = (now or datetime.now(UTC)) - timedelta(days=get_settings().memory_days)
    run("chat_memory", "DELETE FROM agent_sessions WHERE updated_at < :memory_cutoff")
    run("llm_traces", "DELETE FROM llm_calls WHERE at < now() - interval '30 days'")
    run("telegram_chats", "DELETE FROM telegram_chats WHERE created_at < :cutoff AND NOT alerts_on")
    # Keep the newest successful run: the refresh cooldown and skill-report freshness read it.
    run(
        "old_runs",
        """DELETE FROM ingestion_runs WHERE started_at < :cutoff AND id <> COALESCE(
        (SELECT id FROM ingestion_runs WHERE status = 'success' ORDER BY finished_at DESC LIMIT 1), -1)""",
    )
    session.commit()
    return done
