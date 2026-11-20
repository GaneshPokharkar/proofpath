"""The six tools the chat agent can call. Each takes validated arguments and returns plain JSON.

Every job a tool returns is recorded in ctx.seen, so the agent can only cite jobs that really
exist (see app/agent/agent.py: unknown job ids in a reply are stripped and counted).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agent.memory import PROFILE_KEYS, Memory
from app.embeddings import Embedder
from app.ingest.classify import FIELD_LABELS, LEVEL_LABELS, TECH_FIELDS
from app.llm import StructuredLLM
from app.models import Job
from app.search import SearchFilters, hybrid_search

MAX_SHORTLIST = 30


@dataclass
class ToolContext:
    session: Session
    embedder: Embedder
    llm: StructuredLLM | None  # for skill extraction inside explain_match / compare_jobs
    mem: Memory
    seen: dict[int, dict] = field(default_factory=dict)


class ToolError(ValueError):
    """Bad arguments: the message goes back to the model so it can correct itself."""


def _card(job_id: int, title: str, company: str, location, remote, level, apply_by, url) -> dict:
    return {
        "id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "remote": remote,
        "level": level,
        "apply_by": apply_by.isoformat() if apply_by else None,
        "url": url,
    }


def _job(ctx: ToolContext, job_id: Any) -> Job:
    try:
        jid = int(job_id)
    except (TypeError, ValueError):
        raise ToolError(f"job_id must be a number from search results, got {job_id!r}") from None
    job = ctx.session.get(Job, jid)
    if job is None:
        raise ToolError(f"There is no job {jid}. Use ids returned by search_jobs.")
    ctx.seen[job.id] = _card(
        job.id, job.title, job.company, job.location, job.remote, job.level, job.apply_by, job.url
    )
    return job


def _skills(ctx: ToolContext, job_id: int) -> dict[str, list[str]]:
    if ctx.llm is not None:
        from app.market.skills import extract_job_skills

        extract_job_skills(ctx.session, ctx.llm, ctx.embedder, [job_id])  # cached per posting
    rows = ctx.session.execute(
        text(
            "SELECT s.name, js.importance FROM job_skills js JOIN skills s ON s.id = js.skill_id "
            "WHERE js.job_id = :j ORDER BY s.name"
        ),
        {"j": job_id},
    ).all()
    out: dict[str, list[str]] = {"required": [], "preferred": []}
    for name, importance in rows:
        out["required" if importance == "required" else "preferred"].append(name)
    return out


# --- the six tools -----------------------------------------------------------------------------


def search_jobs(
    ctx: ToolContext,
    query: str,
    field: str | None = None,
    level: str | None = None,
    remote: bool | None = None,
    location: str | None = None,
    limit: int = 5,
    india: bool = True,
) -> dict:
    if not query or not str(query).strip():
        raise ToolError("query is required, e.g. 'python backend intern'")
    fields = None
    if field:
        if field == "tech":
            fields = TECH_FIELDS
        elif field in FIELD_LABELS:
            fields = (field,)
        else:
            raise ToolError(f"field must be one of {sorted(FIELD_LABELS)} or 'tech'")
    if level and level not in LEVEL_LABELS:
        raise ToolError(f"level must be one of {sorted(LEVEL_LABELS)}")
    filters = SearchFilters(
        remote=remote,
        location=location or None,
        fields=fields,
        levels=(level,) if level else None,
        india=bool(india),
    )
    hits = hybrid_search(
        ctx.session, ctx.embedder, str(query), filters, limit=max(1, min(int(limit or 5), 10))
    )
    rejected = [r.lower() for r in ctx.mem.profile.get("rejected_roles", [])]
    jobs = []
    for h in hits:
        if any(r and r in h.title.lower() for r in rejected):
            continue
        card = _card(h.job_id, h.title, h.company, h.location, h.remote, h.level, h.apply_by, h.url)
        ctx.seen[h.job_id] = card
        jobs.append(card | {"snippet": h.snippet[:200]})
    return {"count": len(jobs), "jobs": jobs}


def get_job(ctx: ToolContext, job_id: int) -> dict:
    job = _job(ctx, job_id)
    return ctx.seen[job.id] | {"description": job.description[:2500], "open": job.is_active}


def compare_jobs(ctx: ToolContext, job_ids: list) -> dict:
    ids = list(dict.fromkeys(job_ids or []))
    if not 2 <= len(ids) <= 4:
        raise ToolError("Give 2 to 4 job ids to compare")
    rows = []
    for jid in ids:
        job = _job(ctx, jid)
        rows.append(ctx.seen[job.id] | {"skills": _skills(ctx, job.id)})
    have = {s.lower() for s in ctx.mem.profile.get("skills", [])}
    for r in rows:
        req = r["skills"]["required"]
        r["student_has_required"] = [s for s in req if s.lower() in have]
    return {"jobs": rows}


def shortlist(ctx: ToolContext, action: str, job_id: int | None = None) -> dict:
    current = ctx.mem.shortlist
    if action == "list":
        cards = []
        for jid in current:
            try:
                cards.append(ctx.seen[_job(ctx, jid).id])
            except ToolError:  # the posting was removed since it was saved
                continue
        return {"shortlist": cards}
    if job_id is None:
        raise ToolError("job_id is required to add or remove")
    job = _job(ctx, job_id)
    if action == "add":
        if job.id not in current:
            current = (current + [job.id])[-MAX_SHORTLIST:]
    elif action == "remove":
        current = [j for j in current if j != job.id]
    else:
        raise ToolError("action must be add, remove or list")
    ctx.mem.profile["shortlist"] = current
    return {"ok": True, "shortlist_ids": current}


def save_preference(ctx: ToolContext, key: str, value: str) -> dict:
    if key not in PROFILE_KEYS:
        raise ToolError(f"key must be one of {list(PROFILE_KEYS)}")
    value = str(value).strip()
    prof = ctx.mem.profile
    if key == "remote":
        prof[key] = value.lower() in ("true", "yes", "1", "remote", "only remote")
    elif key == "salary_floor":
        prof[key] = value[:60]
    else:  # list-valued: comma separated, de-duplicated, newest last
        items = [v.strip() for v in value.split(",") if v.strip()]
        if key == "levels" and any(i not in LEVEL_LABELS for i in items):
            raise ToolError(f"levels must be from {sorted(LEVEL_LABELS)}")
        if key == "fields" and any(i not in FIELD_LABELS for i in items):
            raise ToolError(f"fields must be from {sorted(FIELD_LABELS)}")
        merged = [x for x in prof.get(key, []) if x.lower() not in {i.lower() for i in items}] + items
        prof[key] = merged[-25:]
    return {"ok": True, "profile": {k: v for k, v in prof.items() if k != "shortlist"}}


def explain_match(ctx: ToolContext, job_id: int) -> dict:
    job = _job(ctx, job_id)
    skills = _skills(ctx, job.id)
    prof = ctx.mem.profile
    have = {s.lower() for s in prof.get("skills", [])}
    req, pref = skills["required"], skills["preferred"]
    matched = [s for s in req + pref if s.lower() in have]
    weights = len(req) + 0.5 * len(pref)
    got = sum(1 for s in req if s.lower() in have) + 0.5 * sum(1 for s in pref if s.lower() in have)
    notes = []
    if (
        prof.get("locations")
        and job.location
        and not any(loc.lower() in job.location.lower() for loc in prof["locations"])
    ):
        notes.append(f"Location {job.location} is not one of the saved locations {prof['locations']}")
    if prof.get("remote") is True and job.remote is False:
        notes.append("Not remote, but the student wants remote work")
    if prof.get("levels") and job.level not in prof["levels"]:
        notes.append(f"Level is {job.level}; saved levels are {prof['levels']}")
    return {
        "job": ctx.seen[job.id],
        "skill_match": round(got / weights, 2) if weights else None,
        "matched_skills": matched,
        "missing_required": [s for s in req if s.lower() not in have],
        "missing_preferred": [s for s in pref if s.lower() not in have],
        "profile_has_skills": bool(have),
        "notes": notes,
    }


TOOLS: dict[str, Callable[..., dict]] = {
    "search_jobs": search_jobs,
    "get_job": get_job,
    "compare_jobs": compare_jobs,
    "shortlist": shortlist,
    "save_preference": save_preference,
    "explain_match": explain_match,
}

_FIELDS = sorted(FIELD_LABELS) + ["tech"]
_LEVELS = sorted(LEVEL_LABELS)

# Function declarations in the OpenAPI subset Gemini accepts.
DECLARATIONS = [
    {
        "name": "search_jobs",
        "description": "Search live job postings. Use for any request to find, list or show jobs.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Role and skills in a few words, e.g. 'react developer'",
                },
                "field": {
                    "type": "STRING",
                    "enum": _FIELDS,
                    "description": "Job field; 'tech' = any tech field",
                },
                "level": {"type": "STRING", "enum": _LEVELS, "description": "Seniority, e.g. intern"},
                "remote": {"type": "BOOLEAN", "description": "true only when the student asks for remote"},
                "location": {"type": "STRING", "description": "City or country the student named"},
                "limit": {"type": "INTEGER", "description": "How many jobs, 1-10 (default 5)"},
                "india": {
                    "type": "BOOLEAN",
                    "description": "Only jobs in India (default true). false only if the student asks for "
                    "jobs abroad, worldwide, or names a place outside India",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_job",
        "description": "Full details of one job: description, location, deadline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"job_id": {"type": "INTEGER"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "compare_jobs",
        "description": "Compare 2-4 jobs side by side: skills each requires, location, level, deadline.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"job_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
            "required": ["job_ids"],
        },
    },
    {
        "name": "shortlist",
        "description": "Add a job to, remove it from, or list the student's saved shortlist.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "enum": ["add", "remove", "list"]},
                "job_id": {"type": "INTEGER", "description": "Needed for add and remove"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "save_preference",
        "description": "Remember something about the student for later turns: their skills, preferred "
        "locations, remote, levels, fields, salary floor, or roles they do not want.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "key": {"type": "STRING", "enum": list(PROFILE_KEYS)},
                "value": {"type": "STRING", "description": "Comma-separated for lists, e.g. 'Python, SQL'"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "explain_match",
        "description": "Explain how well ONE job fits the student's saved profile: matched and missing skills, "
        "location, remote and level fit.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"job_id": {"type": "INTEGER"}},
            "required": ["job_id"],
        },
    },
]
