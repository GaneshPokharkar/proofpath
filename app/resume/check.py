"""Resume check: compare a student's resume with what live postings for a role ask for, and
suggest projects that would fill the most-wanted gaps.

Privacy: the resume is processed in memory and never stored. Contact details (emails, phone
numbers, URLs) are removed before any text is sent to the LLM.
"""

import io
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import StructuredLLM
from app.market.demand import MarketRead

MAX_RESUME_CHARS = 12_000

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(com|in|io|dev|me|org|net)/\S*)", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{8,}\d)(?!\w)")


class ResumeError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages[:6])
        except (PdfReadError, ValueError) as exc:
            raise ResumeError(
                "That PDF couldn't be read. Try exporting it again, or upload a .txt file."
            ) from exc
    elif name.endswith((".txt", ".md", ".tex")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ResumeError("Upload your resume as a PDF or a .txt file.")
    text = text.strip()
    if len(text) < 200:
        raise ResumeError(
            "We couldn't find much text in that file. If it's a scanned image, upload a text-based PDF."
        )
    return text


def redact(text: str) -> str:
    """Remove contact details; the skills and projects are all the check needs."""
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    return _PHONE.sub("[phone]", text)


class ResumeSkill(BaseModel):
    name: str = Field(description="Canonical skill name, e.g. 'PostgreSQL', 'React'")
    where: Literal["experience", "project", "skills_list", "education"] = Field(
        description="Where in the resume the skill shows up; 'skills_list' if only listed, never used"
    )


class ResumeProfile(BaseModel):
    skills: list[ResumeSkill] = Field(max_length=60)
    level: Literal["student", "fresher", "junior", "experienced"]


class ProjectIdea(BaseModel):
    title: str = Field(description="Short project name, 2-5 words")
    pitch: str = Field(description="One sentence: what it does and who it is for")
    skills: list[str] = Field(
        description="The missing skills this project proves", min_length=1, max_length=4
    )
    steps: list[str] = Field(description="4-6 concrete build steps", min_length=3, max_length=6)
    done_when: str = Field(description="A checkable definition of done, e.g. deployed with a public URL")
    weeks: int = Field(ge=1, le=4, description="Realistic part-time build time for a student")
    resume_bullet: str = Field(
        description="A resume bullet to use AFTER building it. No invented numbers: put any metric "
        "the student must measure in [square brackets]."
    )


class ProjectIdeas(BaseModel):
    projects: list[ProjectIdea] = Field(min_length=1, max_length=3)


PROFILE_PROMPT = """You read a resume and list the technical skills it shows.
- Only concrete technical skills (languages, frameworks, databases, cloud, ML, DevOps, tools,
  named engineering concepts). Skip soft skills.
- Use canonical names ("PostgreSQL", "React", "Machine Learning").
- 'where': 'experience' or 'project' when the resume shows the skill being used there;
  'skills_list' when it is only listed; 'education' when only coursework.
- Never add a skill the resume does not mention."""

PROJECTS_PROMPT = """You are a careers mentor for students in India. Suggest portfolio projects
that would make a student's resume a strong match for a target role.
Rules:
- Each project must prove one or more of the MISSING skills, most-wanted first.
- Build on skills the student already has, so the project is achievable.
- Realistic for one student part-time in 1-4 weeks, using free tools and public data or APIs.
- Prefer projects that are useful to real people and can be deployed with a public link.
- Not a clone of a tutorial (no to-do apps, no plain CRUD).
- The resume bullet describes the finished project honestly; any number goes in [brackets]
  because the student must measure it after building."""


@dataclass(frozen=True)
class SkillMatch:
    skill: str
    demand_share: float
    required: int
    status: str  # used | listed | missing


@dataclass(frozen=True)
class ResumeReport:
    role: str
    sample_size: int
    level: str
    score: float  # demand-weighted share of wanted skills the resume shows being used
    skills: list[SkillMatch]
    projects: list[ProjectIdea]


def _norm(name: str) -> str:
    from app.market.skills import _VERSION_SUFFIX, SEED_ALIASES

    stripped = _VERSION_SUFFIX.sub("", name.strip()) or name.strip()  # "Python 3" -> "Python"
    alias = " ".join(stripped.lower().split())
    return SEED_ALIASES.get(alias, alias).lower()


def check_resume(llm: StructuredLLM, resume_text: str, market: MarketRead) -> ResumeReport:
    text = redact(resume_text)[:MAX_RESUME_CHARS]
    profile = llm.extract(PROFILE_PROMPT, f"Resume:\n{text}", ResumeProfile)

    best: dict[str, str] = {}
    rank = {"experience": 2, "project": 2, "education": 1, "skills_list": 1}
    for s in profile.skills:
        key = _norm(s.name)
        status = "used" if rank[s.where] == 2 else "listed"
        if best.get(key) != "used":
            best[key] = status

    matches = [
        SkillMatch(d.skill, d.share, d.required, best.get(_norm(d.skill), "missing")) for d in market.skills
    ]
    total = sum(m.demand_share for m in matches)
    # Listed-but-never-used skills get half credit: recruiters look for where a skill was applied.
    got = sum(m.demand_share * (1.0 if m.status == "used" else 0.5) for m in matches if m.status != "missing")
    score = round(got / total, 3) if total else 0.0

    missing = [m for m in matches if m.status == "missing"][:8]
    have = [m.skill for m in matches if m.status != "missing"] or [s.name for s in profile.skills[:12]]
    projects: list[ProjectIdea] = []
    if missing:
        ask = (
            f"Target role: {market.role}\nStudent level: {profile.level}\n"
            f"Skills the student already shows: {', '.join(have[:15])}\n"
            "Missing skills, most-wanted first (share of live postings asking for it):\n"
            + "\n".join(f"- {m.skill} ({round(m.demand_share * 100)}%)" for m in missing)
            + "\n\nSuggest 3 projects."
        )
        projects = llm.extract(PROJECTS_PROMPT, ask, ProjectIdeas).projects

    return ResumeReport(market.role, market.sample_size, profile.level, score, matches, projects)


# --- One posting: score the resume against a single job description -----------------------------


class ResumeEdit(BaseModel):
    section: str = Field(description="Where to change: e.g. 'Summary', 'Skills', 'Projects: <name>'")
    change: str = Field(description="What to change, in one sentence, tied to this job description")
    example: str = Field(
        description="Example wording the student can adapt. Only restate what the resume already shows; "
        "never invent experience. Any number the student must measure goes in [brackets]."
    )


class ResumeEdits(BaseModel):
    edits: list[ResumeEdit] = Field(min_length=1, max_length=6)


EDITS_PROMPT = """You help a student tailor an honest resume to ONE job description.
Suggest concrete edits that make skills the resume already has clearly visible to this employer:
use the job's own words for things the student really did, move proof of required skills higher,
and turn a bare skills-list entry into a line showing where it was used.
Never claim experience the resume does not show. Skills the student lacks are handled separately."""


def job_market(job_title: str, company: str, job_id: int, skills: list[tuple[str, str, str]]) -> MarketRead:
    """Treat one posting as the 'market': required skills weigh 1, preferred ones 0.5."""
    from app.market.demand import SkillDemand

    return MarketRead(
        role=f"{job_title} at {company}",
        sample_size=1,
        job_ids=[job_id],
        skills=[
            SkillDemand(name, cat, 1, 1 if imp == "required" else 0, 1.0 if imp == "required" else 0.5)
            for name, cat, imp in skills
        ],
    )


def path_to_100(report: ResumeReport) -> list[dict]:
    """Each remaining gap with the points it is worth, biggest first. Fixing all of them is 100."""
    total = sum(m.demand_share for m in report.skills) or 1.0
    steps = []
    for m in report.skills:
        if m.status == "used":
            continue
        credit = 0.5 if m.status == "listed" else 1.0
        steps.append(
            {
                "skill": m.skill,
                "status": m.status,
                "required": bool(m.required),
                "points": round(m.demand_share * credit / total * 100, 1),
                "action": (
                    f"You list {m.skill} but never show it in use. Add a project or job bullet that says where you used it."
                    if m.status == "listed"
                    else f"Learn {m.skill} and prove it with a project (see the ideas below), then add it to your resume."
                ),
            }
        )
    steps.sort(key=lambda s: -s["points"])
    if steps:  # absorb rounding so the steps always add up to exactly 100
        steps[0]["points"] = round(100 - round(report.score * 100) - sum(s["points"] for s in steps[1:]), 1)
    return steps


def suggest_edits(llm: StructuredLLM, resume_text: str, job_title: str, job_text: str) -> list[ResumeEdit]:
    text = redact(resume_text)[:MAX_RESUME_CHARS]
    ask = f"Job: {job_title}\n\nJob description:\n{job_text[:6000]}\n\nResume:\n{text}"
    return llm.extract(EDITS_PROMPT, ask, ResumeEdits).edits
