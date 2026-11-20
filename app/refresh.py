"""Job refresh that is safe to trigger from anywhere (cron, admin page, Telegram).

- A Postgres advisory lock means two triggers never ingest at the same time.
- A cooldown skips a refresh that would follow a successful one too soon (Remotive asks for
  at most 4 fetches a day), unless forced by an admin.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import func, select, text

from app.config import get_settings
from app.db import get_engine, session_scope
from app.embeddings import Embedder
from app.ingest.pipeline import run_ingestion
from app.ingest.registry import load_sources
from app.models import IngestionRun

LOCK_KEY = 720_431  # arbitrary constant identifying the refresh lock


@dataclass
class RefreshResult:
    status: str  # done | skipped_recent | skipped_running
    message: str
    new: int = 0
    changed: int = 0
    closed: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


def last_success(session) -> datetime | None:
    return session.scalar(select(func.max(IngestionRun.finished_at)).where(IngestionRun.status == "success"))


def run_refresh(
    embedder: Embedder,
    force: bool = False,
    sources=None,
    session_factory=session_scope,
    client: httpx.Client | None = None,
) -> RefreshResult:
    settings = get_settings()
    with session_factory() as session:
        last = last_success(session)
    if not force and last and datetime.now(UTC) - last < timedelta(hours=settings.refresh_cooldown_hours):
        ago = int((datetime.now(UTC) - last).total_seconds() // 60)
        return RefreshResult("skipped_recent", f"Jobs were refreshed {ago} min ago, so this run was skipped.")

    with get_engine().connect() as lock_conn:
        if not lock_conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": LOCK_KEY}).scalar():
            return RefreshResult("skipped_running", "A refresh is already running.")
        try:
            start = datetime.now(UTC)
            own_client = client is None
            client = client or httpx.Client(
                timeout=settings.http_timeout_s,
                headers={"User-Agent": settings.user_agent},
                follow_redirects=True,
            )
            try:
                report = run_ingestion(
                    session_factory,
                    sources if sources is not None else load_sources(settings.sources_file),
                    client,
                    embedder,
                    timedelta(days=settings.expire_after_days),
                )
            finally:
                if own_client:
                    client.close()
            seconds = (datetime.now(UTC) - start).total_seconds()
            new = sum(r.inserted for r in report.runs)
            changed = sum(r.updated for r in report.runs)
            closed = sum(r.deactivated for r in report.runs)
            msg = f"{new} new, {changed} updated, {closed} closed across {len(report.runs)} sources"
            if report.failures:
                msg += f"; {len(report.failures)} source(s) failed"
            return RefreshResult("done", msg, new, changed, closed, dict(report.failures), round(seconds, 1))
        finally:
            lock_conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})
