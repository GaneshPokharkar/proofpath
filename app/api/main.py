"""HTTP API and web pages for ProofPath."""

import hashlib
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
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


# --- Readiness: market demand vs. a student's GitHub evidence --------------------------------


class CoverageOut(BaseModel):
    skill: str
    demand_share: float
    required: int
    proven: bool
    repos: list[str]
    details: list[str]


class ReadinessResponse(BaseModel):
    github: str
    role: str
    score: float
    repos_checked: int
    sample_size: int
    skills: list[CoverageOut]


@app.get("/readiness", response_model=ReadinessResponse)
def readiness_endpoint(
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    github: Annotated[str, Query(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")],
    role: Annotated[str, Query(min_length=2, max_length=120)],
    sample: Annotated[int, Query(ge=5, le=40)] = 20,
) -> ReadinessResponse:
    """How ready a GitHub profile looks for a role: each skill employers ask for, and whether
    the student's own (non-fork) repos show evidence of it."""
    import httpx

    from app.evidence.collect import analyze_student, readiness
    from app.evidence.github import GitHubClient, GitHubError

    settings = get_settings()
    with httpx.Client(timeout=20) as http:
        try:
            summary = analyze_student(session, GitHubClient(http, settings.github_token), embedder, github)
        except GitHubError as exc:
            raise HTTPException(502, str(exc)) from exc
    read = market_read(session, llm, embedder, role, sample=sample)
    result = readiness(session, github, read)
    return ReadinessResponse(
        github=github,
        role=role,
        score=result.score,
        repos_checked=summary.repos - summary.forks_skipped,
        sample_size=read.sample_size,
        skills=[CoverageOut(**vars(c)) for c in result.skills],
    )


# --- Refresh: admin page, daily cron, Telegram --------------------------------------------


def _require_admin(request: Request) -> None:
    settings = get_settings()
    if not settings.admin_secret:
        raise HTTPException(503, "Admin is not set up yet: add ADMIN_SECRET in the Vercel settings")
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied, settings.admin_secret):
        raise HTTPException(401, "Wrong admin password")


def _refresh(request: Request, force: bool):
    from app.refresh import run_refresh

    return run_refresh(get_embedder(request), force=force)


@app.post("/admin/refresh")
def admin_refresh(request: Request, force: bool = False) -> dict:
    _require_admin(request)
    return vars(_refresh(request, force))


@app.get("/cron/daily")
def cron_daily(request: Request) -> dict:
    """Called by Vercel Cron once a day: refresh jobs, then send Telegram alerts.

    Vercel sends `Authorization: Bearer $CRON_SECRET` when CRON_SECRET is set. Until it is, the
    call is accepted from Vercel's cron user agent; the refresh cooldown limits any abuse."""
    settings = get_settings()
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    allowed = [s for s in (settings.cron_secret, settings.admin_secret) if s]
    if allowed:
        if not any(secrets.compare_digest(supplied, s) for s in allowed):
            raise HTTPException(401, "Not authorised")
    elif not request.headers.get("user-agent", "").startswith("vercel-cron"):
        raise HTTPException(401, "Not authorised")

    result = vars(_refresh(request, force=False))
    result["alerts_sent"] = _send_alerts(request)
    return result


def _send_alerts(request: Request) -> int:
    from app.db import session_scope
    from app.telegram.bot import TelegramClient, build_alerts

    token = get_settings().telegram_bot_token
    if not token:
        return 0
    client = TelegramClient(token)
    sent = 0
    with session_scope() as session:
        for chat_id, message in build_alerts(session, get_embedder(request)):
            try:
                client.send(chat_id, message)
                sent += 1
            except RuntimeError:
                pass  # a user who blocked the bot must not stop everyone else's alerts
    return sent


@app.get("/admin/status")
def admin_status(session: SessionDep) -> dict:
    """What is configured, and when jobs were last refreshed (no secrets are revealed)."""
    from app.refresh import last_success

    s = get_settings()
    last = last_success(session)
    return {
        "last_refresh": last,
        "configured": {
            "ai_key": bool(s.llm_api_key),
            "admin_password": bool(s.admin_secret),
            "cron_secret": bool(s.cron_secret),
            "telegram_bot": bool(s.telegram_bot_token),
            "telegram_admins": bool(s.telegram_admins),
            "github_token": bool(s.github_token),
        },
    }


# --- Telegram webhook --------------------------------------------------------------------


def _webhook_secret(token: str) -> str:
    # Telegram echoes this in X-Telegram-Bot-Api-Secret-Token, proving a request came from it.
    return hashlib.sha256(f"proofpath:{token}".encode()).hexdigest()[:48]


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> dict:
    from app.db import session_scope
    from app.telegram.bot import Deps, TelegramClient, handle_update

    s = get_settings()
    if not s.telegram_bot_token:
        raise HTTPException(503, "Telegram bot is not set up yet: add TELEGRAM_BOT_TOKEN")
    if request.headers.get("x-telegram-bot-api-secret-token") != _webhook_secret(s.telegram_bot_token):
        raise HTTPException(401, "Not from Telegram")
    update = await request.json()

    llm = None
    if s.llm_api_key:
        llm = getattr(request.app.state, "llm", None) or build_llm(s.llm_provider, s.llm_api_key, s.llm_model)
        request.app.state.llm = llm

    with session_scope() as session:
        deps = Deps(
            session=session,
            embedder_factory=lambda: get_embedder(request),
            admins={a.strip().lstrip("@").lower() for a in s.telegram_admins.split(",") if a.strip()},
            refresh=lambda force: _refresh(request, force),
            market=(lambda role: market_read(session, llm, get_embedder(request), role, sample=20))
            if llm
            else None,
        )
        reply = handle_update(update, deps)
    if reply:
        TelegramClient(s.telegram_bot_token).send(*reply)
    return {"ok": True}  # Telegram retries anything that is not a 200


@app.post("/admin/telegram/setup")
def telegram_setup(request: Request) -> dict:
    """Point the bot's webhook at this deployment (run once after adding TELEGRAM_BOT_TOKEN)."""
    from app.telegram.bot import TelegramClient

    _require_admin(request)
    s = get_settings()
    if not s.telegram_bot_token:
        raise HTTPException(503, "Add TELEGRAM_BOT_TOKEN in the Vercel settings first")
    url = str(request.base_url).rstrip("/").replace("http://", "https://") + "/telegram/webhook"
    client = TelegramClient(s.telegram_bot_token)
    client.call(
        "setWebhook", url=url, secret_token=_webhook_secret(s.telegram_bot_token), allowed_updates=["message"]
    )
    client.call(
        "setMyCommands",
        commands=[
            {"command": "jobs", "description": "Search live jobs"},
            {"command": "skills", "description": "Skills employers ask for in a role"},
            {"command": "alerts", "description": "Daily new-job alerts"},
            {"command": "help", "description": "How to use ProofPath"},
        ],
    )
    return {"webhook": url, "bot": client.call("getMe").get("username")}


# --- Web pages ---------------------------------------------------------------------------

_WEB = Path(__file__).resolve().parent.parent / "web"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return (_WEB / "index.html").read_text()


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_page() -> str:
    return (_WEB / "admin.html").read_text()


# --- Resume check ------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class SkillMatchOut(BaseModel):
    skill: str
    demand_share: float
    required: int
    status: str


class ProjectOut(BaseModel):
    title: str
    pitch: str
    skills: list[str]
    steps: list[str]
    done_when: str
    weeks: int
    resume_bullet: str


class ResumeResponse(BaseModel):
    role: str
    sample_size: int
    level: str
    score: float
    skills: list[SkillMatchOut]
    projects: list[ProjectOut]


@app.post("/resume/check", response_model=ResumeResponse)
async def resume_check(
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    resume: Annotated[UploadFile, File()],
    role: Annotated[str, Form(min_length=2, max_length=120)],
) -> ResumeResponse:
    """Which in-demand skills a resume shows for a role, and projects that would fill the gaps.
    The file is read in memory and never stored; contact details are removed before the LLM."""
    from app.resume.check import ResumeError, check_resume, extract_text

    data = await resume.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 2 MB. Upload a smaller PDF.")
    try:
        text = extract_text(resume.filename or "", data)
    except ResumeError as exc:
        raise HTTPException(422, str(exc)) from exc
    read = market_read(session, llm, embedder, role, sample=20)
    report = check_resume(llm, text, read)
    return ResumeResponse(
        role=report.role,
        sample_size=report.sample_size,
        level=report.level,
        score=report.score,
        skills=[SkillMatchOut(**vars(s)) for s in report.skills],
        projects=[ProjectOut(**p.model_dump()) for p in report.projects],
    )
