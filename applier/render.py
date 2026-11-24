"""Render a tailored packet into the candidate's LaTeX layout and compile a one-page PDF.

Uses the same layout as Ganesh's hand-made resume (applier/templates/classic_preamble.tex): single
column, plain section titles and no kerning tricks, so ATS text extraction reads every word whole.
Compiles with a local pdflatex when present, otherwise with the LaTeX build service ProofPath uses.
If the PDF runs past one page, the least relevant bullet is dropped and it is compiled again.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from app.resume.tailor import compile_pdf, page_count, tex
from applier.profile import Profile
from applier.tailor import Packet

PREAMBLE = (Path(__file__).parent / "templates" / "classic_preamble.tex").read_text(encoding="utf-8")


def _link(url: str, label: str) -> str:
    return rf"\href{{{url}}}{{\underline{{{tex(label)}}}}}"


def render(profile: Profile, p: Packet) -> str:
    c = profile.contact
    head = PREAMBLE.replace("<<TITLE>>", tex(f"{c.name} - Resume - {p.job.title}")).replace(
        "<<NAME>>", tex(c.name)
    )
    head = head.replace("<<ROLE>>", tex(p.job.title)).replace("<<KEYWORDS>>", tex(", ".join(p.covered[:20])))
    links = [
        _link(u, u.removeprefix("https://").removeprefix("http://"))
        for u in (c.github, c.linkedin, c.website)
        if u
    ]
    out = [
        head,
        r"\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}r@{}}",
        rf"  {{\LARGE\textbf{{{tex(c.name)}}}}} & {tex(c.phone)} \\",
        rf"  & \href{{mailto:{c.email}}}{{{tex(c.email)}}} \\",
        rf"  {tex(', '.join(x for x in (c.city, c.state, c.country) if x))} & {' \; '.join(links)} \\",
        r"\end{tabular*}",
        "",
        r"\section{Summary}",
        rf"{{\small {tex(p.summary)}}}",
        "",
        r"\section{Education}",
        r"{\small",
        r"\begin{tabularx}{\textwidth}{|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|c|c|}",
        r"  \hline",
        r"  \textbf{Degree/Examination} & \textbf{Institution/Board} & \textbf{CGPA/Percentage} & \textbf{Year} \\",
        r"  \hline",
    ]
    for e in profile.education:
        out += [rf"  {tex(e.degree)} & {tex(e.institution)} & {tex(e.score)} & {tex(e.year)} \\", r"  \hline"]
    out += [r"\end{tabularx}}", "", r"\section{Projects}"]
    for proj, bullets in p.projects:
        extra = " ".join(f"| {_link(u, k)}" for k, u in proj.links.items())
        title = tex(proj.title).replace(r"\textendash{}", "--")
        out.append(
            rf"\entry{{{title}{(r' \textmd{\small ' + extra + '}') if extra else ''}}}{{{tex(proj.date)}}}{{{tex(', '.join(proj.tech[:8]))}}}"
        )
        out += [r"\begin{points}", *[rf"  \item {tex(b)}" for b in bullets], r"\end{points}", ""]
    if p.experience:
        out.append(r"\section{Internships}")
        for exp, bullets in p.experience:
            out.append(rf"\entry{{{tex(exp.org)}}}{{{tex(exp.dates)}}}{{{tex(exp.role)}}}")
            out += [r"\begin{points}", *[rf"  \item {tex(b)}" for b in bullets], r"\end{points}", ""]
    out += [r"\section{Technical Skills}", r"{\small"]
    out += [rf"\textbf{{{tex(g)}:}} {tex(', '.join(s))}\\" for g, s in p.skills.items() if s]
    out += ["}", ""]
    if profile.achievements:
        out += [
            r"\section{Certifications and Achievements}",
            r"\begin{itemize}[leftmargin=12pt,topsep=1pt,itemsep=0pt,parsep=0pt]\small",
            *[rf"  \item {tex(a)}" for a in profile.achievements],
            r"\end{itemize}",
        ]
    out.append(r"\end{document}")
    return "\n".join(out).replace("--", "--")


def compile_local_or_remote(source: str, service_url: str) -> bytes:
    if shutil.which("pdflatex"):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "r.tex").write_text(source, encoding="utf-8")
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "r.tex"], cwd=d, capture_output=True, timeout=120
                )
            pdf = Path(d) / "r.pdf"
            if pdf.exists():
                return pdf.read_bytes()
    return compile_pdf(source, service_url)


def build_pdf(
    profile: Profile, p: Packet, service_url: str, max_trims: int = 10
) -> tuple[str, bytes, list[str]]:
    """Returns (LaTeX source, PDF bytes, bullets dropped to fit one page)."""
    dropped: list[str] = []
    for _ in range(max_trims + 1):
        source = render(profile, p)
        pdf = compile_local_or_remote(source, service_url)
        if page_count(pdf) <= 1:
            return source, pdf, dropped
        # Least relevant first: the last project's last bullet, keeping at least one per entry.
        for _proj, bullets in reversed(p.projects):
            if len(bullets) > 1:
                dropped.append(bullets.pop())
                break
        else:
            if len(p.projects) > 2:
                proj, bullets = p.projects.pop()
                dropped.append(f"(project) {proj.title}")
            else:
                return source, pdf, dropped
    return source, pdf, dropped
