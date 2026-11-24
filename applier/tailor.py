"""Turn (master profile, job description) into a tailored, fact-checked application packet.

1. read_jd: the LLM pulls out must-haves, nice-to-haves and the job's own keywords.
2. plan: the LLM chooses which projects and experience to show, in what order, and writes bullets
   in the job's language. Every bullet names the profile facts it is based on.
3. verify: each new line goes through the same two checks as ProofPath's ResumeForge: rules (no
   number or job keyword the profile doesn't contain) and an LLM fact-check against the profile.
   A rejected line falls back to the profile's own wording.
4. keywords: the report lists which job keywords the resume covers and which it honestly can't.
5. screen: a strict recruiter-style LLM screen scores the result, like the companies' own screeners.
6. cover letter: written only from profile facts, and checked the same way.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.resume.tailor import VERIFY_PROMPT, Verdicts, rule_check
from applier.profile import Profile

# --- 1. read the job -----------------------------------------------------------------------------


class JobInfo(BaseModel):
    title: str
    company: str
    location: str = ""
    must_have: list[str] = Field(description="Hard requirements: degrees, languages, skills")
    nice_to_have: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(
        description="Short skill and technology terms exactly as the JD writes them, e.g. 'C++', 'Object-Oriented Design', 'testing'"
    )


JD_PROMPT = """Read this job description. Extract the job title, company, location, the hard
requirements, the nice-to-haves, and the short keywords a resume screener would look for, using the
job's own wording."""


def read_jd(llm, jd_text: str) -> JobInfo:
    return llm.extract(JD_PROMPT, jd_text[:12000], JobInfo)


# --- 2. plan the resume --------------------------------------------------------------------------


class BulletPlan(BaseModel):
    text: str = Field(description="One resume bullet, one line where possible, starting with a verb")
    fact_numbers: list[int] = Field(
        description="Numbers of the facts (from the item's list) this bullet is based on"
    )


class ItemPlan(BaseModel):
    id: str = Field(description="Project or experience id from the profile")
    bullets: list[BulletPlan] = Field(min_length=1, max_length=4)


class Plan(BaseModel):
    summary: str = Field(
        description="2-3 full sentences (40-60 words) for this job, using only the summary facts and item facts, naming the strongest projects"
    )
    projects: list[ItemPlan] = Field(description="2-3 most relevant projects, most relevant first")
    experience: list[ItemPlan] = Field(default_factory=list)
    skills_first: list[str] = Field(description="Profile skills to list first because the job asks for them")


PLAN_PROMPT = """You tailor a student's resume to ONE job, using ONLY the facts given.
- Choose the 3 projects that best prove what the job asks for (2 only if there are just 2), most
  relevant first. Include every experience entry.
- For each chosen item, START FROM ITS DEFAULT BULLETS: keep each one, or rewrite it so the job's
  keywords and the most relevant evidence come first. Keep the specifics; never make a bullet
  shorter or vaguer than the default. The first project gets 3-4 bullets, the others 2-3. You may
  add one bullet from that item's facts if it proves a job requirement better.
- Each bullet lists the numbers of the facts it uses. Prefer concrete evidence (built, tested,
  deployed) over adjectives.
- Never add a skill, tool, employer, result, number or responsibility the facts don't contain. If
  the job wants something the facts lack, leave it out; the report will show it as a gap.
- Use the job's exact keyword spelling where the facts support it (e.g. "Object-Oriented Design").
- skills_first: only skills that appear in the profile's skills list."""


def _facts_listing(profile: Profile) -> str:
    out = ["SUMMARY FACTS:", *[f"- {f}" for f in profile.summary_facts], ""]
    for kind, items in (("PROJECT", profile.projects), ("EXPERIENCE", profile.experience)):
        for it in items:
            head = it.title if kind == "PROJECT" else f"{it.role} at {it.org}"
            out.append(f"{kind} id={it.id}: {head}")
            if kind == "PROJECT":
                out.append(f"  tech: {', '.join(it.tech)}")
            out += [f"  fact {i}: {f}" for i, f in enumerate(it.facts)]
            out += [f"  default bullet: {b}" for b in it.bullets]
    out.append("SKILLS: " + "; ".join(f"{g}: {', '.join(s)}" for g, s in profile.skills.items()))
    return "\n".join(out)


# --- 3. verify -----------------------------------------------------------------------------------


@dataclass
class Line:
    where: str  # "summary", project id or experience id
    text: str
    accepted: bool
    reason: str = ""
    fallback: str = ""


def verify(llm, profile: Profile, lines: list[Line], job_keywords: list[str]) -> None:
    source = profile.source_text()
    for ln in lines:
        reason = rule_check(ln.text, source, job_keywords)
        if reason:
            ln.accepted, ln.reason = False, reason
    to_judge = [ln for ln in lines if ln.accepted]
    if not to_judge:
        return
    listing = "\n".join(f"{i}. {ln.text}" for i, ln in enumerate(to_judge))
    verdicts = llm.extract(
        VERIFY_PROMPT, f"ORIGINAL (all true facts):\n{source}\n\nREWRITTEN lines:\n{listing}", Verdicts
    )
    by_index = {v.index: v for v in verdicts.verdicts}
    for i, ln in enumerate(to_judge):
        v = by_index.get(i)
        if v is None or not v.supported:
            ln.accepted, ln.reason = False, (v.reason if v else "no verdict")


# --- 4-6. packet ---------------------------------------------------------------------------------


class Screen(BaseModel):
    score: int = Field(description="0-100: how likely this resume is to be shortlisted")
    decision: str = Field(description="shortlist, maybe or reject")
    met: list[str]
    missing: list[str]
    concerns: list[str]


SCREEN_PROMPT = """You are a strict AI resume screener for this job. Judge only on evidence in
the resume. List requirements met, requirements missing, and real concerns."""


class CoverLetter(BaseModel):
    text: str = Field(description="Under 230 words. Plain paragraphs, no placeholders except [AVAILABILITY]")


COVER_PROMPT = """Write a short, specific cover letter for this job using ONLY the facts given.
Open with the role and job ID if known; one paragraph on the most relevant project with concrete
evidence; one line on why this company; close with availability as the literal text [AVAILABILITY].
No clichés ("I am passionate"), no claims beyond the facts."""


@dataclass
class Packet:
    job: JobInfo
    summary: str
    projects: list[tuple[object, list[str]]]  # (Project, bullets)
    experience: list[tuple[object, list[str]]]
    skills: dict[str, list[str]]
    lines: list[Line]
    covered: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    screen: Screen | None = None
    cover_letter: str = ""


def _ordered_skills(profile: Profile, first: list[str]) -> dict[str, list[str]]:
    want = [s.lower() for s in first]
    out = {}
    for group, items in profile.skills.items():
        out[group] = sorted(
            items, key=lambda s: (s.lower() not in want, want.index(s.lower()) if s.lower() in want else 0)
        )
    return dict(sorted(out.items(), key=lambda kv: not any(s.lower() in want for s in kv[1])))


_GENERIC = {
    "development",
    "developer",
    "architecture",
    "architectures",
    "experience",
    "skills",
    "knowledge",
    "understanding",
    "fundamentals",
    "programming",
    "solutions",
    "and",
    "or",
    "of",
    "the",
    "in",
    "with",
}


def _norm(s: str) -> str:
    s = (
        s.lower()
        .replace("front end", "frontend")
        .replace("back end", "backend")
        .replace("full stack", "full-stack")
    )
    return re.sub(r"\s+", " ", s)


def _has(text: str, term: str) -> bool:
    return re.search(rf"(?<![\w+#]){re.escape(term)}(?![\w+#])", text) is not None


def keyword_coverage(text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """A keyword counts as covered when the resume has it, one of its alternatives ("C/C++" ->
    "C" or "C++"), or every meaningful word of a longer phrase ("full-stack web development")."""
    low = _norm(text)
    covered, gaps = [], []
    for k in dict.fromkeys(keywords):
        kn = _norm(k)
        alternatives = (
            [a.strip() for a in re.split(r"/| or ", kn) if a.strip()] if re.search(r"/| or ", kn) else [kn]
        )
        words = [w for w in re.split(r"[\s,]+", kn) if w and w not in _GENERIC]
        ok = (
            _has(low, kn)
            or any(_has(low, a) for a in alternatives)
            or (bool(words) and all(_has(low, w) for w in words))
        )
        (covered if ok else gaps).append(k)
    return covered, gaps


def build_packet(llm, profile: Profile, jd_text: str) -> Packet:
    job = read_jd(llm, jd_text)
    plan = llm.extract(
        PLAN_PROMPT,
        f"JOB:\n{jd_text[:8000]}\n\nKEYWORDS: {', '.join(job.keywords)}\n\n{_facts_listing(profile)}",
        Plan,
    )

    projects = {p.id: p for p in profile.projects}
    experience = {e.id: e for e in profile.experience}
    lines = [Line("summary", plan.summary.strip(), True, fallback=" ".join(profile.summary_facts[:3]) + ".")]
    chosen: list[tuple[str, object, list[Line]]] = []
    for kind, plans, pool in (
        ("project", plan.projects, projects),
        ("experience", plan.experience, experience),
    ):
        for ip in plans:
            item = pool.get(ip.id)
            if item is None:
                continue
            item_lines = []
            for i, b in enumerate(ip.bullets):
                fb = item.bullets[i] if i < len(item.bullets) else ""
                item_lines.append(Line(ip.id, b.text.strip(), True, fallback=fb))
            chosen.append((kind, item, item_lines))
            lines += item_lines
    # Every experience entry appears even if the plan skipped it, with its own bullets.
    for e in profile.experience:
        if not any(it is e for _, it, _ in chosen):
            chosen.append(("experience", e, [Line(e.id, b, True) for b in e.bullets]))
    if not any(k == "project" for k, _, _ in chosen):
        for p in profile.projects[:3]:
            chosen.append(("project", p, [Line(p.id, b, True) for b in p.bullets]))

    verify(llm, profile, [ln for ln in lines if ln.text], job.keywords)

    def final(ls: list[Line]) -> list[str]:
        return [
            ln.text if ln.accepted else ln.fallback for ln in ls if (ln.text if ln.accepted else ln.fallback)
        ]

    summary = lines[0].text if lines[0].accepted else lines[0].fallback
    packet = Packet(
        job=job,
        summary=summary,
        projects=[(it, final(ls)) for k, it, ls in chosen if k == "project"],
        experience=[(it, final(ls)) for k, it, ls in chosen if k == "experience"],
        skills=_ordered_skills(profile, [s for s in plan.skills_first if s.lower() in profile.all_skills()]),
        lines=lines,
    )
    return packet


def resume_text(profile: Profile, p: Packet) -> str:
    parts = [p.summary]
    for item, bullets in p.projects + p.experience:
        parts += [
            getattr(item, "title", "") or getattr(item, "role", ""),
            ", ".join(getattr(item, "tech", [])),
            *bullets,
        ]
    parts += [f"{g}: {', '.join(s)}" for g, s in p.skills.items()]
    parts += [f"{e.degree} {e.institution}" for e in profile.education] + profile.achievements
    return "\n".join(parts)


def finish(llm, profile: Profile, p: Packet, jd_text: str) -> Packet:
    """Keyword coverage, the recruiter screen and the cover letter, on the final resume text."""
    text = resume_text(profile, p)
    p.covered, p.gaps = keyword_coverage(text, p.job.keywords)
    p.screen = llm.extract(SCREEN_PROMPT, f"JOB:\n{jd_text[:8000]}\n\nRESUME:\n{text}", Screen)
    letter = llm.extract(
        COVER_PROMPT,
        f"JOB:\n{jd_text[:6000]}\n\nCANDIDATE: {profile.contact.name}\n{_facts_listing(profile)}",
        CoverLetter,
    ).text
    reason = rule_check(letter.replace("[AVAILABILITY]", ""), profile.source_text(), p.job.keywords)
    p.cover_letter = letter if reason is None else ""
    if reason:
        p.lines.append(Line("cover_letter", letter, False, reason))
    return p
