"""One-day retention: nothing in the database outlives 24 hours.

Runs from the daily cron. Resumes and API keys are never written at all; this clears the rest.
Job postings are kept only while they were seen by a refresh in the last day, and only when a
refresh actually succeeded in that day, so a failed scrape never empties the site.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

RETENTION = timedelta(days=1)


def purge(session: Session, now: datetime | None = None) -> dict[str, int]:
    cutoff = (now or datetime.now(UTC)) - RETENTION
    p = {"cutoff": cutoff}
    done: dict[str, int] = {}

    def run(name: str, sql: str) -> None:
        done[name] = session.execute(text(sql), p).rowcount or 0

    fresh_run = session.execute(
        text("SELECT 1 FROM ingestion_runs WHERE status = 'success' AND finished_at >= :cutoff LIMIT 1"), p
    ).first()
    run("closed_jobs", "DELETE FROM jobs WHERE NOT is_active")
    if fresh_run:
        run("stale_jobs", "DELETE FROM jobs WHERE last_seen_at < :cutoff")
    run("skill_reports", "DELETE FROM market_cache WHERE built_at < :cutoff")
    run("github_checks", "DELETE FROM students WHERE created_at < :cutoff")
    run("unused_skills", """DELETE FROM skills s WHERE s.created_at < :cutoff
        AND NOT EXISTS (SELECT 1 FROM job_skills js WHERE js.skill_id = s.id)
        AND NOT EXISTS (SELECT 1 FROM repo_evidence re WHERE re.skill_id = s.id)""")
    run("telegram_chats", "DELETE FROM telegram_chats WHERE created_at < :cutoff")
    # Keep the newest successful run: the refresh cooldown and skill-report freshness read it.
    run("old_runs", """DELETE FROM ingestion_runs WHERE started_at < :cutoff AND id <> COALESCE(
        (SELECT id FROM ingestion_runs WHERE status = 'success' ORDER BY finished_at DESC LIMIT 1), -1)""")
    session.commit()
    return done
