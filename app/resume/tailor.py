"""Tailor a resume to one job, verify every rewritten line, and render a one-page PDF.

1. Contact details are pulled out with regexes and never sent to the LLM; they are put back
   only into the LaTeX header.
2. The LLM parses the resume into ResumeDoc (validated JSON).
3. The LLM rewrites each bullet for the job, using the job's words only for things the resume
   already shows.
4. A verifier rejects any rewritten bullet the source resume does not back: a rule check (no new
   numbers outside [brackets], no job skill the resume never mentions) and then an LLM judge.
   A rejected bullet falls back to the original wording.
5. The result is rendered into a LaTeX template, compiled, and if it runs past one page the
   least relevant bullets are trimmed one at a time and it is compiled again.
"""

import base64
import io
import re
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, Field

from app.llm import StructuredLLM
from app.resume.check import MAX_RESUME_CHARS, redact

# --- 1. contact details stay out of the LLM -----------------------------------------------------

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+?\d[\d\s-]{8,}\d)")
LINK = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[\w\-/]+", re.I)


def contacts(text: str) -> dict:
    links = LINK.findall(text)
    return {
        "email": (EMAIL.findall(text) or [None])[0],
        "phone": (PHONE.findall(text) or [None])[0],
        "linkedin": next((u for u in links if "linkedin" in u.lower()), None),
        "github": next((u for u in links if "github" in u.lower()), None),
    }


# --- 2. structured resume ------------------------------------------------------------------------


class Entry(BaseModel):
    title: str = Field(description="Role or project name")
    org: str = Field(default="", description="Company / institution, or the tech stack for a project")
    dates: str = Field(default="")
    bullets: list[str] = Field(default_factory=list, max_length=8)


class EducationRow(BaseModel):
    degree: str
    school: str
    score: str = ""
    year: str = ""


class SkillGroup(BaseModel):
    category: str
    items: list[str]


class ResumeDoc(BaseModel):
    name: str
    location: str = ""
    summary: str = ""
    education: list[EducationRow] = Field(default_factory=list, max_length=6)
    experience: list[Entry] = Field(default_factory=list, max_length=6)
    projects: list[Entry] = Field(default_factory=list, max_length=6)
    skills: list[SkillGroup] = Field(default_factory=list, max_length=8)
    achievements: list[str] = Field(default_factory=list, max_length=6)


PARSE_PROMPT = """Turn this resume into the given JSON structure. Copy text faithfully; do not
improve, shorten or add anything. Bullets go in the same order as in the resume. Leave a field empty
when the resume has nothing for it."""


# --- 3. rewrite for the job ----------------------------------------------------------------------


class Rewrite(BaseModel):
    section: str = Field(description="'experience' or 'projects'")
    entry: int = Field(description="0-based index of the entry in that section")
    bullet: int = Field(description="0-based index of the bullet in that entry")
    text: str = Field(description="The rewritten bullet")


class Rewrites(BaseModel):
    summary: str = Field(
        description="A 1-2 sentence summary aimed at this job, using only facts in the resume"
    )
    bullets: list[Rewrite]
    skills_first: list[str] = Field(
        description="Skills from the resume to list first because the job asks for them"
    )


REWRITE_PROMPT = """You tailor a student's resume to ONE job. Rewrite bullets so the work the
student really did is described in the job's language and the most relevant facts come first.
Hard rules:
- Never add a skill, tool, employer, result or number the resume does not contain.
- Any number the student would have to measure goes in [square brackets].
- Keep each bullet one line where possible, starting with a strong verb.
- Only rewrite bullets that get better for this job; leave the rest out of your answer."""


# --- 4. verifier --------------------------------------------------------------------------------


class Verdict(BaseModel):
    index: int
    supported: bool = Field(description="true only if every claim in the rewrite is backed by the resume")
    reason: str = Field(default="", description="For unsupported lines: the claim the resume does not back")


class Verdicts(BaseModel):
    verdicts: list[Verdict]


VERIFY_PROMPT = """You are a strict fact-checker. For each rewritten resume line, decide whether the
ORIGINAL resume supports every claim in it: the same tools, the same kind of work, the same scale.
Rewording is fine; anything new (a tool, a result, a number, a responsibility) is unsupported.
Text in [square brackets] is a placeholder the student fills in, not a claim."""

NUMBER = re.compile(r"(?<!\[)\b\d[\d,.]*\s*(?:%|x|k|\+)?")


def _numbers(s: str) -> set[str]:
    without_placeholders = re.sub(r"\[[^\]]*\]", "", s)
    return {n.strip().rstrip(".,") for n in NUMBER.findall(without_placeholders)}


def rule_check(rewrite: str, source_text: str, job_skills: list[str]) -> str | None:
    """Cheap deterministic checks before the LLM judge. Returns a reason to reject, or None."""
    src = source_text.lower()
    new_numbers = {n for n in _numbers(rewrite) if n and n.lower() not in src}
    if new_numbers:
        return f"adds numbers not in the resume: {', '.join(sorted(new_numbers))}"
    for skill in job_skills:
        s = skill.lower()
        if re.search(rf"(?<![\w]){re.escape(s)}(?![\w])", rewrite.lower()) and s not in src:
            return f"mentions {skill}, which the resume never shows"
    return None


@dataclass
class Checked:
    section: str
    entry: int
    bullet: int
    original: str
    rewrite: str
    accepted: bool
    reason: str = ""


@dataclass
class Tailored:
    doc: ResumeDoc
    checked: list[Checked]
    coverage_before: float
    coverage_after: float
    job_skills: list[str]
    contact: dict
    tex: str = ""
    pdf: bytes | None = None
    pages: int | None = None
    trimmed: list[str] = field(default_factory=list)
    compile_error: str | None = None


def coverage(doc: ResumeDoc, job_skills: list[str]) -> float:
    """Share of the job's skills that appear anywhere in the resume text."""
    if not job_skills:
        return 0.0
    blob = doc.model_dump_json().lower()
    return round(sum(1 for s in job_skills if s.lower() in blob) / len(job_skills), 3)


def tailor(
    llm: StructuredLLM, resume_text: str, job_title: str, job_text: str, job_skills: list[str]
) -> Tailored:
    contact = contacts(resume_text)
    clean = redact(resume_text)[:MAX_RESUME_CHARS]
    doc = llm.extract(PARSE_PROMPT, f"Resume:\n{clean}", ResumeDoc)
    before = coverage(doc, job_skills)

    rw = llm.extract(
        REWRITE_PROMPT,
        f"Job: {job_title}\nSkills the job asks for: {', '.join(job_skills)}\n\nJob description:\n"
        f"{job_text[:5000]}\n\nResume as JSON:\n{doc.model_dump_json()}",
        Rewrites,
    )

    candidates: list[Checked] = []
    for r in rw.bullets:
        entries = (
            doc.experience if r.section == "experience" else doc.projects if r.section == "projects" else None
        )
        if (
            entries is None
            or not 0 <= r.entry < len(entries)
            or not 0 <= r.bullet < len(entries[r.entry].bullets)
        ):
            continue
        original = entries[r.entry].bullets[r.bullet]
        if r.text.strip() == original.strip():
            continue
        reason = rule_check(r.text, clean, job_skills)
        candidates.append(
            Checked(r.section, r.entry, r.bullet, original, r.text.strip(), reason is None, reason or "")
        )

    summary_reason = rule_check(rw.summary, clean, job_skills) if rw.summary else "empty"
    if summary_reason is None:  # the new summary is fact-checked like any bullet
        candidates.append(Checked("summary", 0, 0, doc.summary, rw.summary.strip(), True))
    to_judge = [c for c in candidates if c.accepted]
    if to_judge:
        listing = "\n".join(f"{i}. {c.rewrite}" for i, c in enumerate(to_judge))
        verdicts = llm.extract(
            VERIFY_PROMPT, f"ORIGINAL resume:\n{clean}\n\nREWRITTEN lines:\n{listing}", Verdicts
        )
        by_index = {v.index: v for v in verdicts.verdicts}
        for i, c in enumerate(to_judge):
            v = by_index.get(i)
            if v is None or not v.supported:
                c.accepted, c.reason = False, (v.reason if v else "the verifier gave no verdict")

    for c in candidates:
        if c.accepted and c.section == "summary":
            doc.summary = c.rewrite
        elif c.accepted:
            entries = doc.experience if c.section == "experience" else doc.projects
            entries[c.entry].bullets[c.bullet] = c.rewrite
    if rw.skills_first:
        first = {s.lower() for s in rw.skills_first if s.lower() in clean.lower()}
        for g in doc.skills:
            g.items.sort(key=lambda s: s.lower() not in first)
        doc.skills.sort(key=lambda g: not any(i.lower() in first for i in g.items))

    return Tailored(doc, candidates, before, coverage(doc, job_skills), job_skills, contact)


# --- 5. LaTeX ------------------------------------------------------------------------------------

_TEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex(s: str) -> str:
    return "".join(_TEX_ESCAPE.get(ch, ch) for ch in (s or "")).replace("–", "--").replace("—", "---")


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[a4paper,top=0.45in,bottom=0.4in,left=0.45in,right=0.45in]{geometry}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{tabularx}
\usepackage{array}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\raggedright
\titleformat{\section}{\vspace{-5pt}\scshape\raggedright\large\bfseries}{}{0em}{}[\vspace{-6pt}\rule{\textwidth}{0.6pt}\vspace{-4pt}]
\titlespacing*{\section}{0pt}{5pt}{1pt}
\newcommand{\entry}[3]{\textbf{#1} \hfill {\small\textit{#2}}\\[-1pt]{\small\textit{#3}}\vspace{-3pt}}
\newenvironment{points}{\begin{itemize}[label=--,leftmargin=14pt,topsep=1pt,itemsep=0pt,parsep=0pt]\small}{\end{itemize}\vspace{1pt}}
\begin{document}
"""


SEP = r" \; "


def render(doc: ResumeDoc, contact: dict) -> str:
    right = [tex(contact[k]) for k in ("phone", "email") if contact.get(k)]
    links = [
        rf"\href{{https://{c.removeprefix('https://').removeprefix('http://')}}}{{\underline{{{label}}}}}"
        for label, c in (("LinkedIn", contact.get("linkedin")), ("GitHub", contact.get("github")))
        if c
    ]
    out = [
        PREAMBLE,
        r"\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}r@{}}",
        rf"{{\LARGE\textbf{{{tex(doc.name)}}}}} & {right[0] if right else ''} \\",
        rf"{tex(doc.location)} & {SEP.join(right[1:] + links)} \\",
        r"\end{tabular*}",
        "",
    ]
    if doc.summary:
        out += [r"\section{Summary}", rf"{{\small {tex(doc.summary)}}}", ""]
    if doc.education:
        out += [
            r"\section{Education}",
            r"{\small\begin{tabularx}{\textwidth}{|X|X|c|c|}\hline",
            r"\textbf{Degree} & \textbf{Institution} & \textbf{Score} & \textbf{Year} \\ \hline",
        ]
        out += [
            rf"{tex(e.degree)} & {tex(e.school)} & {tex(e.score)} & {tex(e.year)} \\ \hline"
            for e in doc.education
        ]
        out += [r"\end{tabularx}}", ""]
    for title, entries in (("Experience", doc.experience), ("Projects", doc.projects)):
        if not entries:
            continue
        out.append(rf"\section{{{title}}}")
        for e in entries:
            out.append(rf"\entry{{{tex(e.title)}}}{{{tex(e.dates)}}}{{{tex(e.org)}}}")
            if e.bullets:
                out += [r"\begin{points}"] + [rf"  \item {tex(b)}" for b in e.bullets] + [r"\end{points}"]
            out.append("")
    if doc.skills:
        out += [r"\section{Skills}", r"{\small"]
        out += [rf"\textbf{{{tex(g.category)}:}} {tex(', '.join(g.items))}\\" for g in doc.skills]
        out += ["}", ""]
    if doc.achievements:
        out += [r"\section{Achievements}", r"\begin{points}"] + [
            rf"  \item {tex(a)}" for a in doc.achievements
        ]
        out += [r"\end{points}"]
    out.append(r"\end{document}")
    return "\n".join(out)


class CompileError(RuntimeError):
    pass


def compile_pdf(source: str, url: str, client: httpx.Client | None = None) -> bytes:
    """Compile with a LaTeX build service (Vercel has no TeX installed)."""
    c = client or httpx.Client(timeout=60)
    r = c.post(url, json={"compiler": "pdflatex", "resources": [{"main": True, "content": source}]})
    if r.status_code >= 300 or not r.content.startswith(b"%PDF"):
        raise CompileError(f"LaTeX build failed ({r.status_code}): {r.text[:300]}")
    return r.content


def page_count(pdf: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf)).pages)


def _relevance(e: Entry, job_skills: list[str]) -> int:
    blob = (e.title + " " + e.org + " " + " ".join(e.bullets)).lower()
    return sum(1 for s in job_skills if s.lower() in blob)


def trim_once(doc: ResumeDoc, job_skills: list[str]) -> str | None:
    """Drop the last bullet of the least job-relevant entry that has more than one bullet."""
    entries = [e for e in doc.experience + doc.projects if len(e.bullets) > 1]
    if not entries:
        if doc.achievements:
            return doc.achievements.pop()
        return None
    victim = min(entries, key=lambda e: (_relevance(e, job_skills), -len(e.bullets)))
    return victim.bullets.pop()


def build_pdf(t: Tailored, url: str, max_trims: int = 8, client: httpx.Client | None = None) -> Tailored:
    for _ in range(max_trims + 1):
        t.tex = render(t.doc, t.contact)
        try:
            t.pdf = compile_pdf(t.tex, url, client)
        except (CompileError, httpx.HTTPError) as exc:
            t.compile_error = str(exc)
            t.pdf = None
            return t
        t.pages = page_count(t.pdf)
        if t.pages <= 1:
            return t
        dropped = trim_once(t.doc, t.job_skills)
        if dropped is None:
            return t
        t.trimmed.append(dropped)
    t.tex = render(t.doc, t.contact)
    return t


def pdf_b64(t: Tailored) -> str | None:
    return base64.b64encode(t.pdf).decode() if t.pdf else None
