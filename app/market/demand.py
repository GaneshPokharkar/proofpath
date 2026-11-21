"""Market read: which skills do live postings for a role actually ask for, and how often."""

import json
import re
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
    llm: StructuredLLM | None,
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


# --- Semantic cache ---------------------------------------------------------------------------
# A saved report is reused for a role that means the same thing. Two layers, because embeddings
# alone were measured to be unsafe here (evals/semantic_cache.py): "java developer" vs
# "javascript developer" scores 0.906, higher than real synonyms like "ml engineer" vs
# "machine learning engineer" (0.891).
#   1. canonical_role(): rewrite spellings and abbreviations, so those match exactly.
#   2. embedding similarity >= SEMANTIC_MIN, and only when the seniority words are the same.

SEMANTIC_MIN = 0.93
_SPELLINGS = [
    (r"\bfront[\s-]+end\b", "frontend"),
    (r"\bback[\s-]+end\b", "backend"),
    (r"\bfull[\s-]+stack\b", "fullstack"),
    (r"\bdev[\s-]+ops\b", "devops"),
    (r"\bml\b", "machine learning"),
    (r"\bai\b", "artificial intelligence"),
    (r"\bsde\b", "software development engineer"),
    (r"\bswe\b", "software engineer"),
    (r"\breact(?:\.?js)\b", "react"),
    (r"\bnode(?:\.?js)\b", "node"),
    (r"\bdevs?\b", "developer"),
    (r"\bengg?\b", "engineer"),
    (r"\bengineering\b(?=\s+(intern|internship|trainee))", "engineer"),
    (r"\binternship\b", "intern"),
]
LEVEL_WORDS = {
    "intern",
    "trainee",
    "junior",
    "senior",
    "lead",
    "principal",
    "staff",
    "head",
    "manager",
    "director",
    "fresher",
}


def canonical_role(role: str) -> str:
    r = " ".join(re.sub(r"[^a-z0-9+#. ]+", " ", role.lower()).split())
    for pattern, repl in _SPELLINGS:
        r = re.sub(pattern, repl, r)
    return " ".join(r.split())


def _levels(role: str) -> set[str]:
    return set(role.split()) & LEVEL_WORDS


def _scope(filters: SearchFilters, sample: int) -> str:
    return f"loc={(filters.location or '').lower()}|remote={filters.remote}|india={filters.india}|n={sample}"


def _cache_key(role: str, filters: SearchFilters, sample: int) -> str:
    return f"{canonical_role(role)}|{_scope(filters, sample)}"


def _semantic_hit(session: Session, embedder: Embedder, role: str, scope: str):
    vec = embedder.embed_query(role)
    row = session.execute(
        text(
            """
            SELECT key, role, payload, 1 - (role_vec <=> CAST(:v AS vector)) AS sim FROM market_cache
            WHERE scope = :s AND role_vec IS NOT NULL AND built_at >= now() - interval '7 days'
            ORDER BY role_vec <=> CAST(:v AS vector) LIMIT 1
            """
        ),
        {"v": str(vec), "s": scope},
    ).first()
    if row is None or row.sim < SEMANTIC_MIN or _levels(canonical_role(row.role)) != _levels(role):
        return None, vec
    return row, vec


class NeedsKey(LookupError):
    """No saved report for this role and no AI key to build one."""


def cached_market_read(
    session: Session,
    llm: StructuredLLM,
    embedder: Embedder,
    role: str,
    filters: SearchFilters = SearchFilters(),
    sample: int = 30,
) -> tuple[MarketRead, bool]:
    """market_read, computed once per role and served from the database to every later request
    for 7 days, so a visitor's own API key is spent only by the first person to ask for a role.

    Returns (read, from_cache)."""
    from app.llmops.trace import CallRecord, timed

    canon = canonical_role(role)
    key, scope = _cache_key(role, filters, sample), _scope(filters, sample)
    hit_rec = CallRecord(task="market_report", provider="cache", model="")
    row = session.execute(
        text(
            """
            SELECT payload FROM market_cache
            WHERE key = :k AND built_at >= now() - interval '7 days'
            """
        ),
        {"k": key},
    ).first()
    vec = None
    if row is None:
        srow, vec = _semantic_hit(session, embedder, canon, scope)
        if srow is not None:
            row, key, hit_rec.provider = srow, srow.key, "semantic-cache"
    if row is not None:
        with timed(hit_rec):
            session.execute(text("UPDATE market_cache SET hits = hits + 1 WHERE key = :k"), {"k": key})
        p = row.payload
        return MarketRead(
            p["role"], p["sample_size"], p["job_ids"], [SkillDemand(**s) for s in p["skills"]]
        ), True

    if llm is None:
        raise NeedsKey(role)
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
            INSERT INTO market_cache (key, role, payload, built_at, scope, role_vec)
            VALUES (:k, :r, CAST(:p AS jsonb), now(), :s, CAST(:v AS vector))
            ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload, built_at = now(),
                role = EXCLUDED.role, scope = EXCLUDED.scope, role_vec = EXCLUDED.role_vec
            """
        ),
        {
            "k": key,
            "r": canon,
            "p": json.dumps(payload),
            "s": scope,
            "v": str(vec or embedder.embed_query(canon)),
        },
    )
    return read, False
