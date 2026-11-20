"""Resume check: text extraction, redaction, scoring and project suggestions (fake LLM)."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.api.main import app
from app.db import get_session
from app.ingest.pipeline import sync_postings
from app.market.demand import MarketRead, SkillDemand
from app.market.skills import ExtractedSkills
from app.resume.check import (
    ProjectIdea,
    ProjectIdeas,
    ResumeEdit,
    ResumeEdits,
    ResumeError,
    ResumeProfile,
    ResumeSkill,
    check_resume,
    extract_text,
    redact,
)
from tests.test_market import FakeLLM as SkillsLLM
from tests.test_pipeline_and_search import FakeSource, posting

RESUME = (
    """Varad Bhoir  varad@example.com  +91 98765 43210  github.com/varad2343
Education: B.Tech E&TC, VIT Pune. Coursework: Machine Learning.
Projects: Smart system - built with Python and Flask, deployed with Docker.
Skills: Python, Flask, Docker, Kubernetes, SQL, Git
"""
    + "Worked on several team projects and hackathons. " * 5
)


class ResumeLLM:
    """Plays both prompts: reads skills from the resume, then writes one project idea."""

    model_name = "fake"

    def __init__(self):
        self.prompts = []

    def extract(self, system, user, schema):
        self.prompts.append(user)
        if schema is ResumeProfile:
            return ResumeProfile(
                level="student",
                skills=[
                    ResumeSkill(name="Python 3", where="project"),
                    ResumeSkill(name="Flask", where="project"),
                    ResumeSkill(name="Docker", where="project"),
                    ResumeSkill(name="k8s", where="skills_list"),
                    ResumeSkill(name="SQL", where="skills_list"),
                ],
            )
        if schema is ProjectIdeas:
            return ProjectIdeas(
                projects=[
                    ProjectIdea(
                        title="Job alert bot",
                        pitch="Sends students new internships daily.",
                        skills=["PostgreSQL", "FastAPI"],
                        steps=["Ingest jobs", "Store in Postgres", "Deploy"],
                        done_when="Deployed with a public link",
                        weeks=2,
                        resume_bullet="Built a job alert bot used by [N] students.",
                    )
                ]
            )
        if schema is ResumeEdits:
            return ResumeEdits(
                edits=[
                    ResumeEdit(
                        section="Skills", change="Show where you used SQL", example="Built [X] with SQL"
                    )
                ]
            )
        return SkillsLLM().extract(system, user, schema)


MARKET = MarketRead(
    role="backend engineer",
    sample_size=10,
    job_ids=[],
    skills=[
        SkillDemand("Python", "language", 9, 9, 0.9),
        SkillDemand("PostgreSQL", "database", 6, 5, 0.6),
        SkillDemand("Kubernetes", "devops", 5, 3, 0.5),
        SkillDemand("Docker", "devops", 4, 2, 0.4),
        SkillDemand("FastAPI", "framework", 3, 2, 0.3),
    ],
)


def test_redact_removes_contact_details():
    out = redact(RESUME)
    assert "varad@example.com" not in out and "98765" not in out and "github.com/varad2343" not in out
    assert "[email]" in out and "[phone]" in out and "Python" in out


def test_extract_text_rules():
    assert "Flask" in extract_text("cv.txt", RESUME.encode())
    with pytest.raises(ResumeError, match="PDF or a .txt"):
        extract_text("cv.docx", b"x")
    with pytest.raises(ResumeError, match="much text"):
        extract_text("cv.txt", b"too short")
    with pytest.raises(ResumeError, match="couldn't be read"):
        extract_text("cv.pdf", b"not a pdf at all")


def test_check_resume_scores_and_suggests_projects():
    llm = ResumeLLM()
    report = check_resume(llm, RESUME, MARKET)
    status = {s.skill: s.status for s in report.skills}
    assert status == {
        "Python": "used",
        "PostgreSQL": "missing",
        "Kubernetes": "listed",
        "Docker": "used",
        "FastAPI": "missing",
    }
    # used = full credit, listed-only = half: (0.9 + 0.4 + 0.5 * 0.5) / 2.7
    assert report.score == pytest.approx((0.9 + 0.4 + 0.25) / 2.7, abs=0.001)
    assert report.projects[0].title == "Job alert bot"
    assert "varad@example.com" not in llm.prompts[0]  # contact details never reach the LLM
    assert "PostgreSQL (60%)" in llm.prompts[1] and "Kubernetes" not in llm.prompts[1].split("Missing")[1]


def test_no_projects_when_nothing_missing():
    market = MarketRead("python dev", 3, [], [SkillDemand("Python", "language", 3, 3, 1.0)])
    assert check_resume(ResumeLLM(), RESUME, market).projects == []


@pytest.fixture
def client(session, embedder):
    jobs = [
        posting("1", "Backend Engineer", "Python, FastAPI and PostgreSQL. Docker."),
        posting("2", "Backend Intern", "Python and Kubernetes."),
    ]
    sync_postings(session, FakeSource(), jobs, embedder, datetime.now(UTC), timedelta(days=14))
    app.state.embedder, app.state.llm = embedder, ResumeLLM()
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.embedder = app.state.llm = None


def test_resume_endpoint(client):
    r = client.post(
        "/resume/check",
        data={"role": "backend engineer"},
        files={"resume": ("cv.txt", RESUME.encode(), "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level"] == "student" and body["projects"][0]["weeks"] == 2
    assert any(s["skill"] == "Python" and s["status"] == "used" for s in body["skills"])


def test_resume_endpoint_rejects_bad_files(client):
    big = client.post(
        "/resume/check",
        data={"role": "x y"},
        files={"resume": ("cv.pdf", b"0" * (2 * 1024 * 1024 + 10), "application/pdf")},
    )
    assert big.status_code == 413
    wrong = client.post(
        "/resume/check", data={"role": "x y"}, files={"resume": ("cv.png", b"png", "image/png")}
    )
    assert wrong.status_code == 422 and "PDF" in wrong.json()["detail"]


def test_unused_import_guard():
    assert ExtractedSkills  # the fake delegates skill extraction to the market fake


def test_gemini_schema_keeps_fields_named_title():
    from app.llm import _inline_refs

    schema = _inline_refs(ProjectIdeas.model_json_schema())
    item = schema["properties"]["projects"]["items"]
    assert "title" in item["properties"] and set(item["required"]) <= set(item["properties"])
    assert "title" not in item  # the annotation is gone


def test_match_resume_to_one_job(client, session):
    job_id = session.execute(text("SELECT id FROM jobs WHERE external_id = '1'")).scalar()
    r = client.post(
        "/resume/match", data={"job_id": job_id}, files={"resume": ("cv.txt", RESUME.encode(), "text/plain")}
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["title"] == "Backend Engineer" and 0 < b["score"] < 1
    # Fixing every listed gap takes the score to exactly 100.
    assert round(b["score"] * 100 + sum(g["points"] for g in b["to_100"])) == 100
    assert b["edits"][0]["section"] == "Skills"


def test_match_unknown_job(client):
    r = client.post(
        "/resume/match", data={"job_id": 999999}, files={"resume": ("cv.txt", RESUME.encode(), "text/plain")}
    )
    assert r.status_code == 404
