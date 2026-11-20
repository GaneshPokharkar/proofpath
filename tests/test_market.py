"""Skill extraction, name normalisation and the market read, with a fake LLM."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.ingest.pipeline import sync_postings
from app.market.demand import market_read
from app.market.skills import ExtractedSkill, ExtractedSkills, SkillResolver, extract_job_skills
from app.models import Job
from app.models_skills import JobSkill, Skill
from tests.test_pipeline_and_search import FakeSource, posting

NOW = datetime(2026, 11, 20, tzinfo=UTC)

# keyword in description -> (name the "model" returns, category, importance)
KEYWORDS = {
    "python": ("Python", "language", "required"),
    "fastapi": ("FastAPI", "framework", "required"),
    "postgres": ("Postgres", "database", "required"),  # deliberately a variant spelling
    "postgresql": ("PostgreSQL", "database", "required"),
    "docker": ("Docker", "devops", "preferred"),
    "kubernetes": ("k8s", "devops", "required"),
    "react": ("React", "framework", "required"),
}


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self):
        self.calls = 0

    def extract(self, system, user, schema):
        assert schema is ExtractedSkills
        self.calls += 1
        words = set(user.lower().replace(",", " ").replace(".", " ").split())
        return ExtractedSkills(
            skills=[
                ExtractedSkill(name=n, category=c, importance=i)
                for kw, (n, c, i) in KEYWORDS.items()
                if kw in words
            ]
        )


JOBS = [
    posting("b1", "Backend Intern", "Python, FastAPI and Postgres. Docker is a plus."),
    posting("b2", "Backend Engineer", "Python and PostgreSQL services on Kubernetes."),
    posting("b3", "Python Developer", "Python and FastAPI APIs, Docker."),
    posting("f1", "Frontend Engineer", "React and TypeScript."),
]


@pytest.fixture
def indexed(session, embedder):
    sync_postings(session, FakeSource(), JOBS, embedder, NOW, timedelta(days=14))
    return session


def job_ids(session):
    return list(session.scalars(select(Job.id).order_by(Job.external_id)))


def test_variant_spellings_resolve_to_one_skill(session, embedder):
    resolver = SkillResolver(session, embedder)
    a = resolver.resolve("Postgres", "database")
    b = resolver.resolve("PostgreSQL", "database")
    c = resolver.resolve("  postgresql ", "database")
    assert a == b == c
    assert session.get(Skill, a).name == "PostgreSQL"
    assert resolver.resolve("k8s", "devops") == resolver.resolve("Kubernetes", "devops")
    assert resolver.resolve("Java", "language") != resolver.resolve("JavaScript", "language")
    # Version numbers are dropped; known abbreviations come from the seed list.
    assert resolver.resolve("Python 3.11", "language") == resolver.resolve("Python", "language")
    assert resolver.resolve("TensorFlow 2.x", "ml") == resolver.resolve("tf", "ml")
    assert resolver.resolve("sklearn", "ml") == resolver.resolve("scikit-learn", "ml")
    assert resolver.resolve("C++", "language") != resolver.resolve("C", "language")


def test_embedding_merge_uses_threshold(session):
    class SameVector:
        model_name = "same"

        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 383 for _ in texts]

    resolver = SkillResolver(session, SameVector())
    first = resolver.resolve("Apache Airflow", "tool")
    # An unseen spelling whose embedding is identical merges into the existing skill...
    assert resolver.resolve("Airflow (Apache)", "tool") == first
    # ...but never across categories.
    assert resolver.resolve("Airflow Conf", "concept") != first


def test_extraction_is_cached_by_content_hash(indexed, embedder):
    llm = FakeLLM()
    ids = job_ids(indexed)
    assert extract_job_skills(indexed, llm, embedder, ids) == 4
    assert extract_job_skills(indexed, llm, embedder, ids) == 0  # nothing changed
    assert llm.calls == 4

    edited = [posting("b1", "Backend Intern", "Python, FastAPI, Postgres and React."), *JOBS[1:]]
    sync_postings(indexed, FakeSource(), edited, embedder, NOW + timedelta(hours=1), timedelta(days=14))
    assert extract_job_skills(indexed, llm, embedder, ids) == 1  # only the edited posting
    b1 = indexed.scalars(select(Job.id).where(Job.external_id == "b1")).one()
    names = set(
        indexed.scalars(
            select(Skill.name).join(JobSkill, JobSkill.skill_id == Skill.id).where(JobSkill.job_id == b1)
        )
    )
    assert names == {"Python", "FastAPI", "PostgreSQL", "React"}  # Docker gone, React added


def test_market_read_ranks_skills_by_demand(indexed, embedder):
    read = market_read(indexed, FakeLLM(), embedder, "python backend", sample=3, min_share=0.0)
    assert read.sample_size == 3
    by_name = {s.skill: s for s in read.skills}
    assert read.skills[0].skill == "Python" and by_name["Python"].jobs == 3
    assert by_name["Python"].share == 1.0
    assert by_name["PostgreSQL"].jobs == 2  # "Postgres" and "PostgreSQL" counted as one
    assert by_name["Docker"].required == 0  # only ever "a plus"
    assert indexed.scalar(select(func.count()).select_from(Skill).where(Skill.name == "Postgres")) == 0


def test_market_read_with_no_matches(indexed, embedder):
    from app.search import SearchFilters

    read = market_read(indexed, FakeLLM(), embedder, "python", SearchFilters(company="nobody"))
    assert read.sample_size == 0 and read.skills == []


def test_gemini_client_parses_and_retries_invalid_json():
    import httpx

    from app.llm import GeminiLLM

    replies = iter(
        [
            '{"skills": [{"name": "Python", "category": "language", "importance": "must"}]}',
            '{"skills": [{"name": "Python", "category": "language", "importance": "required"}]}',
        ]
    )
    sent = []

    def handler(request):
        sent.append(request)
        text = next(replies)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})

    llm = GeminiLLM("k", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = llm.extract("sys", "job", ExtractedSkills)
    assert out.skills[0].importance == "required"
    assert len(sent) == 2  # first answer failed validation and was retried with the error
    body = sent[0].read().decode()
    assert "$ref" not in body and '"responseSchema"' in body
