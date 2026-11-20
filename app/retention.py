"""Retention: personal data lives at most a day; public job data only while someone uses it.

Runs from the daily cron. Resumes and API keys are never written at all; this clears the rest.
Job postings: only jobs someone actually searched are kept. A searched job stays until its
apply-by date passes or it has gone unsearched for SEARCHED_KEEP. Jobs nobody searched are dropped
once a newer refresh has replaced them (only after a refresh succeeded that day, so a failed scrape
never empties the site). The daily scrape brings fresh postings back in.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

RETENTION = timedelta(days=1)
SEARCHED_KEEP = timedelta(days=14)
REPORT_KEEP = timedelta(days=7)


def purge(session: Session, now: datetime | None = None) -> dict[str, int]:
    cutoff = (now or datetime.now(UTC)) - RETENTION
    p = {"cutoff": cutoff, "searched_cutoff": (now or datetime.now(UTC)) - SEARCHED_KEEP}
    done: dict[str, int] = {}

    def run(name: str, sql: str) -> None:
        done[name] = session.execute(text(sql), p).rowcount or 0

    fresh_run = session.execute(
        text("SELECT 1 FROM ingestion_runs WHERE status = 'success' AND finished_at >= :cutoff LIMIT 1"), p
    ).first()
    run("closed_jobs", "DELETE FROM jobs WHERE NOT is_active OR apply_by < CURRENT_DATE")
    run("searched_long_ago", "DELETE FROM jobs WHERE last_served_at < :searched_cutoff")
    if fresh_run:
        run("never_searched", "DELETE FROM jobs WHERE last_served_at IS NULL AND last_seen_at < :cutoff")
    # Skill reports are reused by everyone asking the same role, so nobody's key pays twice.
    p["report_cutoff"] = (now or datetime.now(UTC)) - REPORT_KEEP
    run("skill_reports", "DELETE FROM market_cache WHERE built_at < :report_cutoff")
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
