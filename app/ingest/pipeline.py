"""Ingestion: fetch a source, upsert its jobs, re-embed only what changed, retire closed jobs.

Each source runs in its own transaction and records an ingestion_runs row, so one broken
board never blocks the others and every failure is visible (status=failed + the error).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.ingest.base import JobPosting, Source
from app.ingest.classify import classify_field, classify_level
from app.models import IngestionRun, Job, JobEmbedding

log = logging.getLogger(__name__)


@dataclass
class RunStats:
    source: str
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deactivated: int = 0
    embedded: int = 0


@dataclass
class IngestReport:
    runs: list[RunStats] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def _dedupe(postings: list[JobPosting]) -> list[JobPosting]:
    """Some feeds repeat a posting across pages; keep the first copy of each external_id."""
    seen: set[str] = set()
    unique = []
    for p in postings:
        if p.external_id not in seen:
            seen.add(p.external_id)
            unique.append(p)
    return unique


def _row(posting: JobPosting, source_key: str, now: datetime) -> dict:
    return {
        "source": source_key,
        "external_id": posting.external_id,
        "title": posting.title,
        "company": posting.company,
        "location": posting.location,
        "remote": posting.remote,
        "employment_type": posting.employment_type,
        "tags": list(posting.tags),
        "salary_text": posting.salary_text,
        "description": posting.description,
        "url": posting.url,
        "posted_at": posting.posted_at,
        "first_seen_at": now,
        "last_seen_at": now,
        "content_hash": posting.content_hash(),
        "is_active": True,
        "field": classify_field(posting.title),
        "level": classify_level(posting.title),
    }


def sync_postings(
    session: Session,
    source: Source,
    postings: list[JobPosting],
    embedder: Embedder,
    now: datetime,
    expire_after: timedelta,
) -> RunStats:
    """Write one source's fetched postings to the database. Pure DB logic, no HTTP: easy to test."""
    postings = _dedupe(postings)
    stats = RunStats(source=source.key, fetched=len(postings))

    existing = {
        ext_id: (job_id, content_hash)
        for job_id, ext_id, content_hash in session.execute(
            select(Job.id, Job.external_id, Job.content_hash).where(Job.source == source.key)
        )
    }
    if source.full_snapshot and not postings and existing:
        # An empty response from a board that had jobs is far more likely an API glitch than
        # every role closing at once; retiring everything would wipe the source from search.
        raise RuntimeError(
            f"{source.key} returned 0 jobs but {len(existing)} are stored; refusing to retire them all"
        )

    to_embed: list[tuple[int, JobPosting]] = []
    seen_ids: list[int] = []
    for posting in postings:
        row = _row(posting, source.key, now)
        current = existing.get(posting.external_id)
        if current is None:
            job_id = session.execute(pg_insert(Job).values(**row).returning(Job.id)).scalar_one()
            stats.inserted += 1
            to_embed.append((job_id, posting))
        else:
            job_id, old_hash = current
            if old_hash != row["content_hash"]:
                del row["first_seen_at"]
                session.execute(update(Job).where(Job.id == job_id).values(**row))
                stats.updated += 1
                to_embed.append((job_id, posting))
            else:
                stats.unchanged += 1
        seen_ids.append(job_id)

    # Unchanged jobs: just mark them seen (and reactivate any that had been retired).
    if seen_ids:
        session.execute(update(Job).where(Job.id.in_(seen_ids)).values(last_seen_at=now, is_active=True))

    # Retire jobs that have closed.
    retire = update(Job).where(Job.source == source.key, Job.is_active.is_(True))
    if source.full_snapshot:
        if seen_ids:
            retire = retire.where(Job.id.not_in(seen_ids))
    else:
        retire = retire.where(Job.last_seen_at < now - expire_after)
    stats.deactivated = session.execute(retire.values(is_active=False)).rowcount

    stats.embedded = _embed(session, to_embed, embedder)
    return stats


def _embed(session: Session, items: list[tuple[int, JobPosting]], embedder: Embedder) -> int:
    if not items:
        return 0
    vectors = embedder.embed_documents([p.embedding_text() for _, p in items])
    title_vectors = embedder.embed_documents([p.title for _, p in items])
    rows = [
        {
            "job_id": job_id,
            "model": embedder.model_name,
            "content_hash": posting.content_hash(),
            "embedding": vector,
            "title_embedding": title_vector,
        }
        for (job_id, posting), vector, title_vector in zip(items, vectors, title_vectors, strict=True)
    ]
    stmt = pg_insert(JobEmbedding).values(rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[JobEmbedding.job_id],
            set_={
                "model": stmt.excluded.model,
                "content_hash": stmt.excluded.content_hash,
                "embedding": stmt.excluded.embedding,
                "title_embedding": stmt.excluded.title_embedding,
                "created_at": datetime.now(UTC),
            },
        )
    )
    return len(rows)


def run_ingestion(
    session_factory,
    sources: list[Source],
    client: httpx.Client,
    embedder: Embedder,
    expire_after: timedelta,
) -> IngestReport:
    """Ingest every source. A failing source is recorded and reported, and the rest still run."""
    report = IngestReport()
    for source in sources:
        started = datetime.now(UTC)
        with session_factory() as session:
            run = IngestionRun(source=source.key, started_at=started, status="running")
            session.add(run)
            session.flush()
            run_id = run.id
        try:
            postings = source.fetch(client)
            with session_factory() as session:
                stats = sync_postings(session, source, postings, embedder, started, expire_after)
                session.execute(
                    update(IngestionRun)
                    .where(IngestionRun.id == run_id)
                    .values(
                        status="success",
                        finished_at=datetime.now(UTC),
                        fetched=stats.fetched,
                        inserted=stats.inserted,
                        updated=stats.updated,
                        unchanged=stats.unchanged,
                        deactivated=stats.deactivated,
                        embedded=stats.embedded,
                    )
                )
            report.runs.append(stats)
            log.info("ingested %s: %s", source.key, stats)
        except Exception as exc:  # recorded per source, then surfaced in the report
            message = f"{type(exc).__name__}: {exc}"
            with session_factory() as session:
                session.execute(
                    update(IngestionRun)
                    .where(IngestionRun.id == run_id)
                    .values(status="failed", finished_at=datetime.now(UTC), error=message[:2000])
                )
            report.failures[source.key] = message
            log.error("ingestion failed for %s: %s", source.key, message)
    return report
