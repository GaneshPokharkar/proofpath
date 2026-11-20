"""Hybrid job search: Postgres full-text search and pgvector similarity, each on the title
alone and on the whole posting, merged with reciprocal rank fusion (RRF).

Why hybrid: full-text search nails exact terms ("Kubernetes", "FastAPI", a company name)
but misses paraphrases; embeddings catch meaning ("backend intern" ~ "software engineering
internship, APIs") but blur exact terms. RRF combines the two ranked lists using only
ranks, so the two retrievers' incomparable scores never need to be calibrated:
    score(job) = sum over retrievers of 1 / (k + rank)
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embeddings import Embedder


@dataclass(frozen=True)
class SearchFilters:
    remote: bool | None = None
    location: str | None = None  # case-insensitive substring, e.g. "Bengaluru", "India"
    company: str | None = None  # case-insensitive substring
    posted_within_days: int | None = None
    sources: tuple[str, ...] | None = None  # source key prefixes, e.g. ("greenhouse", "lever:cred")
    exclude_sources: tuple[str, ...] = field(default_factory=tuple)
    fields: tuple[str, ...] | None = None  # e.g. ("software", "data_ai"); see app/ingest/classify.py
    levels: tuple[str, ...] | None = None  # e.g. ("intern", "entry")


@dataclass(frozen=True)
class SearchHit:
    job_id: int
    score: float
    keyword_rank: int | None  # best rank across the two full-text lists; None if neither found it
    semantic_rank: int | None  # best rank across the two vector lists
    title: str
    company: str
    location: str | None
    remote: bool | None
    url: str
    source: str
    posted_at: datetime | None
    snippet: str
    field: str = "other"
    level: str = "mid"


def _filter_sql(filters: SearchFilters, params: dict) -> str:
    """WHERE fragments for the jobs table aliased `j`. Values always go in as bound parameters."""
    clauses = ["j.is_active"]
    if filters.remote is not None:
        clauses.append("j.remote = :f_remote")
        params["f_remote"] = filters.remote
    if filters.location:
        clauses.append("j.location ILIKE :f_location")
        params["f_location"] = f"%{filters.location}%"
    if filters.company:
        clauses.append("j.company ILIKE :f_company")
        params["f_company"] = f"%{filters.company}%"
    if filters.posted_within_days is not None:
        clauses.append("j.posted_at >= now() - make_interval(days => :f_days)")
        params["f_days"] = filters.posted_within_days
    if filters.sources:
        parts = []
        for i, prefix in enumerate(filters.sources):
            parts.append(f"(j.source = :f_src{i} OR j.source LIKE :f_srcp{i})")
            params[f"f_src{i}"] = prefix
            params[f"f_srcp{i}"] = f"{prefix}:%"
        clauses.append("(" + " OR ".join(parts) + ")")
    for i, prefix in enumerate(filters.exclude_sources):
        clauses.append(f"NOT (j.source = :f_xsrc{i} OR j.source LIKE :f_xsrcp{i})")
        params[f"f_xsrc{i}"] = prefix
        params[f"f_xsrcp{i}"] = f"{prefix}:%"
    if filters.fields:
        clauses.append("j.field = ANY(:f_fields)")
        params["f_fields"] = list(filters.fields)
    if filters.levels:
        clauses.append("j.level = ANY(:f_levels)")
        params["f_levels"] = list(filters.levels)
    return " AND ".join(clauses)


def hybrid_search(
    session: Session,
    embedder: Embedder,
    query: str,
    filters: SearchFilters = SearchFilters(),
    limit: int = 20,
    candidates: int = 100,
    rrf_k: int = 60,
) -> list[SearchHit]:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    params: dict = {
        "q": query,
        "qvec": str(embedder.embed_query(query)),
        "k": candidates,
        "rrf_k": rrf_k,
        "limit": limit,
    }
    where = _filter_sql(filters, params)

    # Four retrievers, each returning its top `candidates` jobs, fused with RRF:
    #   title_kw  full-text match on the title alone
    #   body_kw   full-text match on title + company + tags + description (title weighted highest)
    #   title_sem embedding similarity of the query to the title alone
    #   body_sem  embedding similarity of the query to the whole posting
    # Title-only lists matter because many descriptions open with long company boilerplate,
    # which otherwise drowns out what the job actually is.
    #
    # plainto_tsquery ANDs every word, which is too strict for a chat query ("python backend
    # intern remote"); rewriting & to | ORs them, and ts_rank_cd still ranks jobs matching more
    # of the words higher. Normalisation 1 divides by log(document length) so long descriptions
    # do not win just by repeating terms.
    sql = text(
        f"""
        WITH tsq AS (
            SELECT NULLIF(replace(plainto_tsquery('english', :q)::text, '&', '|'), '')::tsquery AS q
        ),
        title_kw AS (
            SELECT j.id, row_number() OVER (ORDER BY ts_rank_cd(j.title_tsv, tsq.q, 1) DESC, j.id) AS rnk
            FROM jobs j, tsq
            WHERE tsq.q IS NOT NULL AND j.title_tsv @@ tsq.q AND {where}
            ORDER BY rnk LIMIT :k
        ),
        body_kw AS (
            SELECT j.id, row_number() OVER (ORDER BY ts_rank_cd(j.search_tsv, tsq.q, 1) DESC, j.id) AS rnk
            FROM jobs j, tsq
            WHERE tsq.q IS NOT NULL AND j.search_tsv @@ tsq.q AND {where}
            ORDER BY rnk LIMIT :k
        ),
        title_sem AS (
            SELECT j.id,
                   row_number() OVER (ORDER BY e.title_embedding <=> CAST(:qvec AS vector), j.id) AS rnk
            FROM job_embeddings e JOIN jobs j ON j.id = e.job_id
            WHERE {where}
            ORDER BY rnk LIMIT :k
        ),
        body_sem AS (
            SELECT j.id, row_number() OVER (ORDER BY e.embedding <=> CAST(:qvec AS vector), j.id) AS rnk
            FROM job_embeddings e JOIN jobs j ON j.id = e.job_id
            WHERE {where}
            ORDER BY rnk LIMIT :k
        ),
        fused AS (
            SELECT id,
                   sum(1.0 / (:rrf_k + rnk)) AS score,
                   min(rnk) FILTER (WHERE kind = 'kw') AS keyword_rank,
                   min(rnk) FILTER (WHERE kind = 'sem') AS semantic_rank
            FROM (
                SELECT id, rnk, 'kw' AS kind FROM title_kw
                UNION ALL SELECT id, rnk, 'kw' FROM body_kw
                UNION ALL SELECT id, rnk, 'sem' FROM title_sem
                UNION ALL SELECT id, rnk, 'sem' FROM body_sem
            ) ranked
            GROUP BY id
        )
        SELECT f.id, f.score, f.keyword_rank, f.semantic_rank,
               j.title, j.company, j.location, j.remote, j.url, j.source, j.posted_at,
               j.field, j.level, left(j.description, 280) AS snippet
        FROM fused f
        JOIN jobs j ON j.id = f.id
        ORDER BY f.score DESC, j.posted_at DESC NULLS LAST, f.id
        LIMIT :limit
        """
    )
    # More HNSW candidates than the default 40, so filtered vector queries still fill `candidates`.
    session.execute(text("SET LOCAL hnsw.ef_search = 200"))
    rows = session.execute(sql, params).all()
    return [
        SearchHit(
            job_id=r.id,
            score=float(r.score),
            keyword_rank=r.keyword_rank,
            semantic_rank=r.semantic_rank,
            title=r.title,
            company=r.company,
            location=r.location,
            remote=r.remote,
            url=r.url,
            source=r.source,
            posted_at=r.posted_at,
            snippet=" ".join(r.snippet.split()),
            field=r.field,
            level=r.level,
        )
        for r in rows
    ]
