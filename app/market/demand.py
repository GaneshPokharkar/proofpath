"""Market read: which skills do live postings for a role actually ask for, and how often."""

import json
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.llm import StructuredLLM
from app.market.skills import extract_job_skills
from app.search import SearchFilters, hybrid_search


@dataclass(frozen=True)
class SkillDemand:
    skill: str
    category: str
    jobs: int  # postings in the sample that ask for it (required or preferred)
    required: int  # of those, postings that list it as a must-have
    share: float  # jobs / sample size


@dataclass(frozen=True)
class MarketRead:
    role: str
    sample_size: int
    job_ids: list[int]
    skills: list[SkillDemand]


def market_read(
    session: Session,
    llm: StructuredLLM,
    embedder: Embedder,
    role: str,
    filters: SearchFilters = SearchFilters(),
    sample: int = 30,
    min_share: float = 0.1,
) -> MarketRead:
    """Pull the `sample` most relevant live postings for `role`, make sure each has its skills
    extracted (cached by content hash), and rank skills by how many postings ask for them."""
    hits = hybrid_search(session, embedder, role, filters, limit=sample)
    job_ids = [h.job_id for h in hits]
    if not job_ids:
        return MarketRead(role=role, sample_size=0, job_ids=[], skills=[])

    extract_job_skills(session, llm, embedder, job_ids)

    rows = session.execute(
        text(
            """
            SELECT s.name, s.category,
                   count(*) AS jobs,
                   count(*) FILTER (WHERE js.importance = 'required') AS required
            FROM job_skills js
            JOIN skills s ON s.id = js.skill_id
            WHERE js.job_id = ANY(:ids)
            GROUP BY s.id, s.name, s.category
            ORDER BY jobs DESC, required DESC, s.name
            """
        ),
        {"ids": job_ids},
    ).all()
    n = len(job_ids)
    skills = [
        SkillDemand(r.name, r.category, r.jobs, r.required, round(r.jobs / n, 3))
        for r in rows
        if r.jobs / n >= min_share
    ]
    return MarketRead(role=role, sample_size=n, job_ids=job_ids, skills=skills)


def _cache_key(role: str, filters: SearchFilters, sample: int) -> str:
    words = " ".join(role.lower().split())
    return f"{words}|loc={(filters.location or '').lower()}|remote={filters.remote}|n={sample}"


def cached_market_read(
    session: Session,
    llm: StructuredLLM,
    embedder: Embedder,
    role: str,
    filters: SearchFilters = SearchFilters(),
    sample: int = 30,
) -> tuple[MarketRead, bool]:
    """market_read, computed once per role and served from the database to every later request
    until the next successful job refresh changes the underlying postings.

    Returns (read, from_cache)."""
    key = _cache_key(role, filters, sample)
    row = session.execute(
        text(
            """
            SELECT payload FROM market_cache
            WHERE key = :k AND built_at >= COALESCE(
                (SELECT max(finished_at) FROM ingestion_runs WHERE status = 'success'), '-infinity')
            """
        ),
        {"k": key},
    ).first()
    if row is not None:
        session.execute(text("UPDATE market_cache SET hits = hits + 1 WHERE key = :k"), {"k": key})
        p = row.payload
        return MarketRead(
            p["role"], p["sample_size"], p["job_ids"], [SkillDemand(**s) for s in p["skills"]]
        ), True

    read = market_read(session, llm, embedder, role, filters, sample=sample, min_share=0.1)
    payload = {
        "role": read.role,
        "sample_size": read.sample_size,
        "job_ids": read.job_ids,
        "skills": [vars(s) for s in read.skills],
    }
    session.execute(
        text(
            """
            INSERT INTO market_cache (key, role, payload, built_at) VALUES (:k, :r, CAST(:p AS jsonb), now())
            ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload, built_at = now(), role = EXCLUDED.role
            """
        ),
        {"k": key, "r": role, "p": json.dumps(payload)},
    )
    return read, False
