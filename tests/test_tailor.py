import io

import httpx
from pypdf import PdfWriter

from app.resume.tailor import (
    Entry,
    ResumeDoc,
    Rewrite,
    Rewrites,
    SkillGroup,
    Verdict,
    Verdicts,
    build_pdf,
    render,
    rule_check,
    tailor,
    tex,
)

RESUME = (
    """Asha Kumar  asha.k@example.com  +91 98765 43210  github.com/ashak
Pune, India
Projects
Sales dashboard (Python, pandas, Power BI) 2026
- Cleaned 3 years of sales CSV data with pandas
- Built charts in Power BI for the college fest team
Marks tracker (Flask, PostgreSQL) 2025
- Made a Flask app to record marks
Skills: Python, SQL, Excel, Git
"""
    + "Coursework in statistics and databases. " * 4
)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def extract(self, system, user, schema):
        self.prompts.append(user)
        if schema is ResumeDoc:
            return ResumeDoc(
                name="Asha Kumar",
                location="Pune, India",
                projects=[
                    Entry(
                        title="Sales dashboard",
                        org="Python, pandas, Power BI",
                        dates="2026",
                        bullets=[
                            "Cleaned 3 years of sales CSV data with pandas",
                            "Built charts in Power BI for the college fest team",
                        ],
                    ),
                    Entry(
                        title="Marks tracker",
                        org="Flask, PostgreSQL",
                        dates="2025",
                        bullets=["Made a Flask app to record marks"],
                    ),
                ],
                skills=[SkillGroup(category="Skills", items=["Git", "Excel", "SQL", "Python"])],
            )
        if schema is Rewrites:
            return Rewrites(
                summary="Student who analyses sales data with Python and SQL.",
                bullets=[
                    Rewrite(
                        section="projects",
                        entry=0,
                        bullet=0,
                        text="Cleaned and analysed 3 years of sales data in Python (pandas)",
                    ),
                    Rewrite(
                        section="projects",
                        entry=0,
                        bullet=1,
                        text="Built Tableau dashboards used by 500 students",
                    ),  # invented
                    Rewrite(
                        section="projects",
                        entry=1,
                        bullet=0,
                        text="Built a Flask and PostgreSQL app that stores student marks",
                    ),
                    Rewrite(section="projects", entry=9, bullet=0, text="out of range, ignored"),
                ],
                skills_first=["SQL", "Python", "Tableau"],
            )
        if schema is Verdicts:
            # judge only sees lines that passed the rule check: index 0 and 1 here
            return Verdicts(
                verdicts=[
                    Verdict(index=0, supported=True),
                    Verdict(index=1, supported=False, reason="'stores' is fine but scale unclear"),
                ]
            )
        raise AssertionError(schema)


def test_contacts_never_reach_the_llm():
    llm = FakeLLM()
    t = tailor(llm, RESUME, "Data Analyst", "SQL, Python, Tableau", ["SQL", "Python", "Tableau"])
    assert t.contact["email"] == "asha.k@example.com" and t.contact["github"] == "github.com/ashak"
    assert all("asha.k@example.com" not in p and "98765" not in p for p in llm.prompts)


def test_verifier_keeps_backed_lines_and_rejects_the_rest():
    t = tailor(FakeLLM(), RESUME, "Data Analyst", "SQL, Python, Tableau", ["SQL", "Python", "Tableau"])
    by_text = {c.rewrite: c for c in t.checked}
    assert by_text["Cleaned and analysed 3 years of sales data in Python (pandas)"].accepted
    invented = by_text["Built Tableau dashboards used by 500 students"]
    assert not invented.accepted and "500" in invented.reason  # the rule check catches it first
    assert not by_text["Built a Flask and PostgreSQL app that stores student marks"].accepted  # LLM judge
    p = t.doc.projects
    assert p[0].bullets[0].startswith("Cleaned and analysed")
    assert p[0].bullets[1] == "Built charts in Power BI for the college fest team"  # original kept
    assert t.doc.skills[0].items[:2] == ["SQL", "Python"]  # job skills first; Tableau not added
    assert "Tableau" not in t.doc.model_dump_json()


def test_rule_check():
    src = "built a flask app with python. 3 years of data"
    assert rule_check("Built a Flask app in Python", src, ["Python"]) is None
    assert "numbers" in rule_check("Served 10k users", src, [])
    assert rule_check("Served [N] users over 3 years", src, []) is None  # placeholders are fine
    assert "Docker" in rule_check("Deployed with Docker", src, ["Docker"])


def test_render_escapes_latex():
    assert tex("R&D 100% #1 a_b") == r"R\&D 100\% \#1 a\_b"
    doc = ResumeDoc(name="A & B", projects=[Entry(title="X", bullets=["50% faster"])])
    source = render(doc, {"email": "a@b.co"})
    assert r"A \& B" in source and r"50\% faster" in source and source.rstrip().endswith(r"\end{document}")


def _pdf(pages: int) -> bytes:
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_build_pdf_trims_until_one_page():
    t = tailor(FakeLLM(), RESUME, "Data Analyst", "SQL", ["SQL", "Python", "Tableau"])
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=_pdf(2 if len(calls) < 2 else 1))

    t = build_pdf(
        t, "https://latex.example/build", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert t.pages == 1 and len(t.trimmed) == 1 and len(calls) == 2
    assert t.compile_error is None and t.pdf.startswith(b"%PDF")


def test_build_pdf_reports_compile_errors():
    t = tailor(FakeLLM(), RESUME, "Data Analyst", "SQL", ["SQL"])
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(400, text="! Undefined control sequence"))
    )
    t = build_pdf(t, "https://latex.example/build", client=client)
    assert t.pdf is None and "Undefined control sequence" in t.compile_error and t.tex


def test_summary_is_fact_checked_too():
    t = tailor(FakeLLM(), RESUME, "Data Analyst", "SQL", ["SQL", "Python", "Tableau"])
    summary = next(c for c in t.checked if c.section == "summary")
    assert not summary.accepted and t.doc.summary == ""
