"""HTTP API and web pages for ProofPath."""

import hashlib
import secrets
import time
from contextlib import asynccontextmanager, suppress
from datetime import date, datetime
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
from app.llm import LLMError, StructuredLLM, build_llm
from app.market.demand import NeedsKey, cached_market_read
from app.models import IngestionRun, Job
from app.search import SearchFilters, hybrid_search


@asynccontextmanager
async def _lifespan(_app):
    # Apply the small additive schema steps on start, so a deploy never runs ahead of the database.
    from app.db import get_engine
    from app.schema_live import ensure_schema

    with suppress(Exception):  # never block start-up; each step reports through its own endpoint
        ensure_schema(get_engine())
    # Connect the Telegram bot by itself once its token is in Vercel: no admin step needed.
    import os

    token, host = get_settings().telegram_bot_token, os.environ.get("VERCEL_PROJECT_PRODUCTION_URL")
    if token and host:
        with suppress(Exception):  # the admin page can still connect it by hand
            _connect_bot(token, f"https://{host}")
    yield


app = FastAPI(title="ProofPath", version="0.2.0", lifespan=_lifespan)


@app.exception_handler(LLMError)
async def _llm_error(_request: Request, exc: LLMError):
    from fastapi.responses import JSONResponse

    if "429" in str(exc) or "quota" in str(exc).lower():
        return JSONResponse(
            status_code=429,
            content={
                "detail": "The free AI limit for today is used up. Add your own free Gemini key above, or try again tomorrow."
            },
        )
    return JSONResponse(
        status_code=502, content={"detail": "The AI service didn't answer. Try again in a minute."}
    )


@app.exception_handler(NeedsKey)
async def _needs_key(_request: Request, _exc: NeedsKey):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content={"detail": "Add your Gemini key to analyse this role"})


SessionDep = Annotated[Session, Depends(get_session)]


def get_embedder(request: Request) -> Embedder:
    # Loaded on first use, not at startup: on serverless hosts a cold start that only serves
    # /health should not pay for loading (or downloading) the embedding model. Tests set it first.
    if getattr(request.app.state, "embedder", None) is None:
        request.app.state.embedder = FastEmbedEmbedder(get_settings().embedding_model)
    return request.app.state.embedder


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def _count(session: Session, event: str) -> None:
    """Anonymous daily usage count; never lets a counting problem break the request."""
    from app.usage import count

    try:
        with session.begin_nested():
            count(session, event)
    except Exception:  # noqa: BLE001
        pass


def get_llm(request: Request) -> StructuredLLM:
    # Bring-your-own-key: a visitor's own Gemini key, sent per request and never stored.
    user_key = (request.headers.get("x-gemini-key") or "").strip()
    if user_key:
        if len(user_key) > 200:
            raise HTTPException(400, "That API key doesn't look right")
        return build_llm("gemini", user_key, get_settings().llm_model)
    if getattr(request.app.state, "llm", None) is None:
        settings = get_settings()
        if not settings.llm_api_key:
            raise HTTPException(503, "LLM not configured: set JOBPILOT_LLM_API_KEY")
        request.app.state.llm = build_llm(settings.llm_provider, settings.llm_api_key, settings.llm_model)
    return request.app.state.llm


LLMDep = Annotated[StructuredLLM, Depends(get_llm)]


def get_optional_llm(request: Request) -> StructuredLLM | None:
    """Like get_llm, but None when there is no key: saved reports still work for everyone."""
    try:
        return get_llm(request)
    except HTTPException as exc:
        if exc.status_code == 503:
            return None
        raise


OptionalLLMDep = Annotated[StructuredLLM | None, Depends(get_optional_llm)]


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
    field: str
    level: str
    apply_by: date | None
    last_seen_at: datetime | None


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


def _expand_fields(field: list[str] | None) -> tuple[str, ...] | None:
    from app.ingest.classify import FIELD_LABELS, TECH_FIELDS

    if not field:
        return None
    out: list[str] = []
    for f in field:
        if f == "tech":
            out.extend(TECH_FIELDS)
        elif f in FIELD_LABELS:
            out.append(f)
        else:
            raise HTTPException(422, f"Unknown field {f!r}")
    return tuple(dict.fromkeys(out))


@app.get("/fields")
def fields(session: SessionDep) -> dict:
    """Fields and levels students can filter by, with how many open jobs each has."""
    from app.ingest.classify import FIELD_LABELS, LEVEL_LABELS

    counts = dict(
        session.execute(text("SELECT field, count(*) FROM jobs WHERE is_active GROUP BY field")).all()
    )
    levels = dict(
        session.execute(text("SELECT level, count(*) FROM jobs WHERE is_active GROUP BY level")).all()
    )
    return {
        "fields": [{"key": k, "label": v, "jobs": counts.get(k, 0)} for k, v in FIELD_LABELS.items()],
        "levels": [{"key": k, "label": v, "jobs": levels.get(k, 0)} for k, v in LEVEL_LABELS.items()],
    }


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
    field: Annotated[
        list[str] | None,
        Query(
            description="software, data_ai, devops_cloud, security, mobile, "
            "frontend, qa, design, product, other; or 'tech' for all but other"
        ),
    ] = None,
    level: Annotated[list[str] | None, Query(description="intern, entry, mid, senior")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    india: bool = False,
) -> SearchResponse:
    settings = get_settings()
    filters = SearchFilters(
        india=india,
        remote=remote,
        location=location,
        company=company,
        posted_within_days=posted_within_days,
        sources=tuple(source) if source else None,
        fields=_expand_fields(field),
        levels=tuple(level) if level else None,
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
    _count(session, "search")
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
                field=h.field,
                level=h.level,
                apply_by=h.apply_by,
                last_seen_at=h.last_seen_at,
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
        "usage_last_30_days": _usage(session),
    }


def _usage(session: Session) -> dict:
    from app.usage import summary

    try:
        with session.begin_nested():
            return summary(session)
    except Exception:  # noqa: BLE001 - before the usage table exists
        return {}


class SkillDemandOut(BaseModel):
    skill: str
    category: str
    jobs: int
    required: int
    share: float


class MarketResponse(BaseModel):
    role: str
    sample_size: int
    cached: bool  # served from the saved result for this role, not recomputed
    skills: list[SkillDemandOut]


@app.get("/market", response_model=MarketResponse)
def market(
    session: SessionDep,
    embedder: EmbedderDep,
    llm: OptionalLLMDep,
    role: Annotated[str, Query(min_length=2, max_length=120)],
    location: str | None = None,
    remote: bool | None = None,
    sample: Annotated[int, Query(ge=5, le=60)] = 30,
) -> MarketResponse:
    """Skills that live postings for `role` ask for, ranked by how many postings ask."""
    read, cached = cached_market_read(
        session, llm, embedder, role, SearchFilters(remote=remote, location=location), sample=sample
    )
    return MarketResponse(
        role=read.role,
        sample_size=read.sample_size,
        cached=cached,
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
    llm: OptionalLLMDep,
    github: Annotated[str, Query(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")],
    role: Annotated[str, Query(min_length=2, max_length=120)],
    sample: Annotated[int, Query(ge=5, le=40)] = 30,
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
    read, _ = cached_market_read(session, llm, embedder, role, sample=sample)
    result = readiness(session, github, read)
    _count(session, "github_check")
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
    from app.db import session_scope
    from app.retention import purge

    with session_scope() as session:
        result["purged"] = purge(session)
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
            market=(lambda role: cached_market_read(session, llm, get_embedder(request), role, sample=30)[0]),
        )
        reply = handle_update(update, deps)
    if reply:
        TelegramClient(s.telegram_bot_token).send(*reply)
    return {"ok": True}  # Telegram retries anything that is not a 200


@app.post("/admin/telegram/setup")
def telegram_setup(request: Request) -> dict:
    """Point the bot's webhook at this deployment (run once after adding TELEGRAM_BOT_TOKEN)."""

    _require_admin(request)
    s = get_settings()
    if not s.telegram_bot_token:
        raise HTTPException(503, "Add TELEGRAM_BOT_TOKEN in the Vercel settings first")
    return _connect_bot(s.telegram_bot_token, str(request.base_url))


def _connect_bot(token: str, base_url: str) -> dict:
    from app.telegram.bot import TelegramClient

    url = base_url.rstrip("/").replace("http://", "https://") + "/telegram/webhook"
    client = TelegramClient(token)
    if client.call("getWebhookInfo").get("url") != url:
        client.call("setWebhook", url=url, secret_token=_webhook_secret(token), allowed_updates=["message"])
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


# --- Chat agent --------------------------------------------------------------------------


class ChatIn(BaseModel):
    session_id: str
    message: str


class ChatOut(BaseModel):
    reply: str
    jobs: list[dict]
    tools: list[str]


def get_chat_model(request: Request):
    """GeminiChat from the visitor's own key, else the server key. 503 when there is neither."""
    from app.llm import GeminiChat

    s = get_settings()
    key = (request.headers.get("x-gemini-key") or "").strip() or s.llm_api_key
    if not key:
        raise HTTPException(503, "Add your Gemini key to chat")
    return GeminiChat(key, s.llm_model)


@app.post("/chat", response_model=ChatOut)
def chat_endpoint(
    body: ChatIn,
    session: SessionDep,
    embedder: EmbedderDep,
    model: Annotated[object, Depends(get_chat_model)],
) -> ChatOut:
    """One turn with the job-search agent. Memory is kept under the anonymous session id."""
    from app.agent.agent import chat
    from app.agent.memory import SESSION_ID
    from app.usage import count

    if not SESSION_ID.match(body.session_id):
        raise HTTPException(422, "Bad session id")
    message = body.message.strip()
    if not 1 <= len(message) <= 1000:
        raise HTTPException(422, "Write a message of up to 1000 characters")
    result = chat(model, model, session, embedder, body.session_id, message)
    count(session, "chat")
    session.commit()
    return ChatOut(reply=result.reply, jobs=result.jobs, tools=[c["name"] for c in result.tool_calls])


@app.get("/chat/{session_id}/memory")
def chat_memory(session_id: str, session: SessionDep) -> dict:
    """What the agent remembers for this browser: the profile, shortlist and summary."""
    from app.agent.memory import SESSION_ID, load

    if not SESSION_ID.match(session_id):
        raise HTTPException(422, "Bad session id")
    mem = load(session, session_id)
    return {
        "profile": {k: v for k, v in mem.profile.items() if k != "shortlist"},
        "shortlist": mem.shortlist,
        "summary": mem.summary,
        "turns": mem.turns,
    }


@app.delete("/chat/{session_id}")
def chat_forget(session_id: str, session: SessionDep) -> dict:
    from app.agent.memory import SESSION_ID, forget

    if not SESSION_ID.match(session_id):
        raise HTTPException(422, "Bad session id")
    forget(session, session_id)
    session.commit()
    return {"forgotten": True}


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


class GapOut(BaseModel):
    skill: str
    status: str
    required: bool
    points: float
    action: str


class EditOut(BaseModel):
    section: str
    change: str
    example: str


class JobMatchResponse(BaseModel):
    job_id: int
    title: str
    company: str
    location: str | None
    url: str
    apply_by: date | None
    level: str
    score: float  # 0..1 against this one job description
    skills: list[SkillMatchOut]
    to_100: list[GapOut]  # every remaining gap and the points it is worth; all of them = 100
    edits: list[EditOut]
    projects: list[ProjectOut]


@app.post("/resume/match", response_model=JobMatchResponse)
async def resume_match(
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    resume: Annotated[UploadFile, File()],
    job_id: Annotated[int, Form(ge=1)],
) -> JobMatchResponse:
    """Score a resume against ONE posting and list exactly what would take it to 100.
    The file is read in memory and never stored; contact details are removed before the LLM."""
    from app.market.skills import extract_job_skills
    from app.resume.check import (
        ResumeError,
        check_resume,
        extract_text,
        job_market,
        path_to_100,
        suggest_edits,
    )

    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "That job is no longer listed. Search again and pick another.")
    data = await resume.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 2 MB. Upload a smaller PDF.")
    try:
        text_ = extract_text(resume.filename or "", data)
    except ResumeError as exc:
        raise HTTPException(422, str(exc)) from exc

    extract_job_skills(session, llm, embedder, [job_id])  # once per posting, shared with everyone
    rows = session.execute(
        text(
            """SELECT s.name, s.category, js.importance FROM job_skills js JOIN skills s ON s.id = js.skill_id
               WHERE js.job_id = :j ORDER BY js.importance = 'required' DESC, s.name"""
        ),
        {"j": job_id},
    ).all()
    if not rows:
        raise HTTPException(422, "This posting doesn't name any technical skills to compare against.")
    report = check_resume(llm, text_, job_market(job.title, job.company, job_id, [tuple(r) for r in rows]))
    edits = suggest_edits(llm, text_, job.title, job.description)
    _count(session, "resume_match")
    return JobMatchResponse(
        job_id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        url=job.url,
        apply_by=job.apply_by,
        level=report.level,
        score=report.score,
        skills=[SkillMatchOut(**vars(m)) for m in report.skills],
        to_100=[GapOut(**g) for g in path_to_100(report)],
        edits=[EditOut(**e.model_dump()) for e in edits],
        projects=[ProjectOut(**p.model_dump()) for p in report.projects],
    )


class CheckedOut(BaseModel):
    section: str
    original: str
    rewrite: str
    accepted: bool
    reason: str


class TailorResponse(BaseModel):
    job_id: int
    title: str
    company: str
    coverage_before: float  # share of the job's skills visible in the resume, before / after
    coverage_after: float
    lines: list[CheckedOut]  # every rewrite and the verifier's decision
    trimmed: list[str]  # bullets dropped to fit one page
    pages: int | None
    tex: str
    pdf_base64: str | None
    compile_error: str | None


@app.post("/resume/tailor", response_model=TailorResponse)
async def resume_tailor(
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    resume: Annotated[UploadFile, File()],
    job_id: Annotated[int, Form(ge=1)],
) -> TailorResponse:
    """Rewrite a resume for one job, reject any line the original doesn't back, and return a
    one-page PDF plus its LaTeX. The file is read in memory and never stored."""
    from app.market.skills import extract_job_skills
    from app.resume.check import ResumeError, extract_text
    from app.resume.tailor import build_pdf, pdf_b64, tailor

    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "That job is no longer listed. Search again and pick another.")
    data = await resume.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 2 MB. Upload a smaller PDF.")
    try:
        text_ = extract_text(resume.filename or "", data)
    except ResumeError as exc:
        raise HTTPException(422, str(exc)) from exc
    extract_job_skills(session, llm, embedder, [job_id])
    skills = [
        r[0]
        for r in session.execute(
            text("SELECT s.name FROM job_skills js JOIN skills s ON s.id = js.skill_id WHERE js.job_id = :j"),
            {"j": job_id},
        ).all()
    ]
    t = tailor(llm, text_, job.title, job.description, skills)
    t = build_pdf(t, get_settings().latex_compile_url)
    _count(session, "resume_tailor")
    return TailorResponse(
        job_id=job.id,
        title=job.title,
        company=job.company,
        coverage_before=t.coverage_before,
        coverage_after=t.coverage_after,
        lines=[
            CheckedOut(
                section=c.section,
                original=c.original,
                rewrite=c.rewrite,
                accepted=c.accepted,
                reason=c.reason,
            )
            for c in t.checked
        ],
        trimmed=t.trimmed,
        pages=t.pages,
        tex=t.tex,
        pdf_base64=pdf_b64(t),
        compile_error=t.compile_error,
    )


class ProfileSkillIn(BaseModel):
    name: str
    where: str


class ProfileIn(BaseModel):
    skills: list[ProfileSkillIn]
    level: str = "student"


@app.post("/resume/profile")
async def resume_profile(llm: LLMDep, resume: Annotated[UploadFile, File()]) -> dict:
    """Read a resume once into its skills (and where each is shown). Nothing is stored: the browser
    keeps this and sends it with /match/scores, so scoring many jobs needs no more AI calls."""
    from app.resume.check import ResumeError, extract_text, read_profile

    data = await resume.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "That file is over 2 MB. Upload a smaller PDF.")
    try:
        text_ = extract_text(resume.filename or "", data)
    except ResumeError as exc:
        raise HTTPException(422, str(exc)) from exc
    return read_profile(llm, text_).model_dump()


class ScoresIn(BaseModel):
    profile: ProfileIn
    job_ids: list[int]


@app.post("/match/scores")
def match_scores(body: ScoresIn, session: SessionDep, embedder: EmbedderDep, llm: OptionalLLMDep) -> dict:
    """Match score for each job (up to 20) plus the changes that would take it to 100.
    A posting's skills are read by the AI once and reused for everyone; postings not read yet are
    read with the visitor's key, or reported as pending when there is none."""
    from app.market.skills import extract_job_skills
    from app.resume.check import ResumeProfile, profile_statuses, score_job

    ids = list(dict.fromkeys(body.job_ids))[:20]
    if not ids:
        return {"scores": {}}
    if llm is not None:
        extract_job_skills(session, llm, embedder, ids)
    rows = session.execute(
        text(
            """SELECT js.job_id, s.name, js.importance FROM job_skills js JOIN skills s ON s.id = js.skill_id
               WHERE js.job_id = ANY(:ids) ORDER BY js.importance = 'required' DESC, s.name"""
        ),
        {"ids": ids},
    ).all()
    done = set(
        session.scalars(
            text("SELECT job_id FROM job_skill_extractions WHERE job_id = ANY(:ids)"), {"ids": ids}
        )
    )
    per_job: dict[int, list[tuple[str, str]]] = {}
    for job_id, name, importance in rows:
        per_job.setdefault(job_id, []).append((name, importance))
    from pydantic import ValidationError

    try:
        profile = ResumeProfile.model_validate(body.profile.model_dump())
    except ValidationError as exc:
        raise HTTPException(422, "Upload your resume again") from exc
    statuses = profile_statuses(profile)
    out = {}
    for jid in ids:
        if jid not in done:
            out[jid] = {"pending": True}
        else:
            out[jid] = score_job(statuses, per_job.get(jid, []))
    _count(session, "match_scores")
    return {"scores": out}


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
    read, _ = cached_market_read(session, llm, embedder, role, sample=30)
    report = check_resume(llm, text, read)
    _count(session, "resume_check")
    return ResumeResponse(
        role=report.role,
        sample_size=report.sample_size,
        level=report.level,
        score=report.score,
        skills=[SkillMatchOut(**vars(s)) for s in report.skills],
        projects=[ProjectOut(**p.model_dump()) for p in report.projects],
    )
