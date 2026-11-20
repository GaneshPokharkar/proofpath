"""Market read: which skills do live postings for a role actually ask for, and how often."""

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
