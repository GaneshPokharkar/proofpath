"""HTTP API. Phase 1 exposes search; the chat agent arrives in phase 2."""

import time
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.embeddings import Embedder, FastEmbedEmbedder
from app.llm import StructuredLLM, build_llm
from app.market.demand import market_read
from app.models import IngestionRun, Job
from app.search import SearchFilters, hybrid_search

app = FastAPI(title="ProofPath", version="0.2.0")

SessionDep = Annotated[Session, Depends(get_session)]


def get_embedder(request: Request) -> Embedder:
    # Loaded on first use, not at startup: on serverless hosts a cold start that only serves
    # /health should not pay for loading (or downloading) the embedding model. Tests set it first.
    if getattr(request.app.state, "embedder", None) is None:
        request.app.state.embedder = FastEmbedEmbedder(get_settings().embedding_model)
    return request.app.state.embedder


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def get_llm(request: Request) -> StructuredLLM:
    if getattr(request.app.state, "llm", None) is None:
        settings = get_settings()
        if not settings.llm_api_key:
            raise HTTPException(503, "LLM not configured: set JOBPILOT_LLM_API_KEY")
        request.app.state.llm = build_llm(settings.llm_provider, settings.llm_api_key, settings.llm_model)
    return request.app.state.llm


LLMDep = Annotated[StructuredLLM, Depends(get_llm)]


class JobHit(BaseModel):
    id: int
    title: str
    company: str
    location: str | None
    remote: bool | None
    url: str
    source: str
    posted_at: datetime | None
    snippet: str
    score: float
    keyword_rank: int | None
    semantic_rank: int | None


class SearchResponse(BaseModel):
    query: str
    count: int
    took_ms: float
    results: list[JobHit]


class JobDetail(BaseModel):
    id: int
    title: str
    company: str
    location: str | None
    remote: bool | None
    employment_type: str | None
    tags: list[str]
    salary_text: str | None
    description: str
    url: str
    source: str
    posted_at: datetime | None
    is_active: bool


@app.get("/health")
def health(session: SessionDep) -> dict:
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(
    session: SessionDep,
    embedder: EmbedderDep,
    q: Annotated[str, Query(min_length=1, max_length=300)],
    remote: bool | None = None,
    location: str | None = None,
    company: str | None = None,
    posted_within_days: Annotated[int | None, Query(ge=1, le=365)] = None,
    source: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SearchResponse:
    settings = get_settings()
    filters = SearchFilters(
        remote=remote,
        location=location,
        company=company,
        posted_within_days=posted_within_days,
        sources=tuple(source) if source else None,
    )
    start = time.perf_counter()
    hits = hybrid_search(
        session,
        embedder,
        q,
        filters,
        limit=limit,
        candidates=settings.search_candidates,
        rrf_k=settings.rrf_k,
    )
    took_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(
        query=q,
        count=len(hits),
        took_ms=round(took_ms, 1),
        results=[
            JobHit(
                id=h.job_id,
                title=h.title,
                company=h.company,
                location=h.location,
                remote=h.remote,
                url=h.url,
                source=h.source,
                posted_at=h.posted_at,
                snippet=h.snippet,
                score=round(h.score, 5),
                keyword_rank=h.keyword_rank,
                semantic_rank=h.semantic_rank,
            )
            for h in hits
        ],
    )


@app.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: int, session: SessionDep) -> JobDetail:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return JobDetail(
        id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        remote=job.remote,
        employment_type=job.employment_type,
        tags=job.tags,
        salary_text=job.salary_text,
        description=job.description,
        url=job.url,
        source=job.source,
        posted_at=job.posted_at,
        is_active=job.is_active,
    )


@app.get("/stats")
def stats(session: SessionDep) -> dict:
    """Index size per source and each source's latest ingestion run."""
    per_source = session.execute(
        select(Job.source, func.count().filter(Job.is_active), func.count())
        .group_by(Job.source)
        .order_by(Job.source)
    ).all()
    latest = (
        select(IngestionRun)
        .ext(distinct_on(IngestionRun.source))
        .order_by(IngestionRun.source, IngestionRun.started_at.desc())
    )
    runs = {r.source: r for r in session.scalars(latest)}
    return {
        "active_jobs": sum(active for _, active, _ in per_source),
        "sources": [
            {
                "source": src,
                "active": active,
                "total": total,
                "last_run": {
                    "status": runs[src].status,
                    "started_at": runs[src].started_at,
                    "inserted": runs[src].inserted,
                    "updated": runs[src].updated,
                    "deactivated": runs[src].deactivated,
                    "error": runs[src].error,
                }
                if src in runs
                else None,
            }
            for src, active, total in per_source
        ],
    }


class SkillDemandOut(BaseModel):
    skill: str
    category: str
    jobs: int
    required: int
    share: float


class MarketResponse(BaseModel):
    role: str
    sample_size: int
    skills: list[SkillDemandOut]


@app.get("/market", response_model=MarketResponse)
def market(
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    role: Annotated[str, Query(min_length=2, max_length=120)],
    location: str | None = None,
    remote: bool | None = None,
    sample: Annotated[int, Query(ge=5, le=60)] = 30,
) -> MarketResponse:
    """Skills that live postings for `role` ask for, ranked by how many postings ask."""
    read = market_read(
        session, llm, embedder, role, SearchFilters(remote=remote, location=location), sample=sample
    )
    return MarketResponse(
        role=read.role,
        sample_size=read.sample_size,
        skills=[SkillDemandOut(**vars(s)) for s in read.skills],
    )
