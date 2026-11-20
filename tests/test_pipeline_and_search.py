"""Ingestion pipeline and hybrid search against a real Postgres + pgvector database."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from app.ingest.base import JobPosting
from app.ingest.pipeline import sync_postings
from app.models import Job, JobEmbedding
from app.search import SearchFilters, hybrid_search

NOW = datetime(2026, 11, 20, 6, 0, tzinfo=UTC)
EXPIRE = timedelta(days=14)


@dataclass
class FakeSource:
    key: str = "greenhouse:acme"
    full_snapshot: bool = True
    postings: list[JobPosting] = field(default_factory=list)

    def fetch(self, client):
        return self.postings


def posting(ext_id: str, title: str, description: str, **kw) -> JobPosting:
    return JobPosting(
        external_id=ext_id,
        title=title,
        company=kw.pop("company", "Acme"),
        url=f"https://jobs.example.com/{ext_id}",
        description=description,
        **kw,
    )


CORPUS = [
    posting(
        "1",
        "Backend Engineering Intern",
        "Build Python FastAPI services and PostgreSQL APIs.",
        location="Bengaluru, India",
        remote=False,
    ),
    posting(
        "2",
        "Machine Learning Intern",
        "Work on LLM agents, embeddings and retrieval with Python.",
        location="Remote - India",
        remote=True,
    ),
    posting(
        "3",
        "Senior Sales Manager",
        "Own enterprise accounts and quotas across EMEA.",
        location="London",
        remote=False,
    ),
    posting(
        "4",
        "DevOps Engineer",
        "Terraform, Kubernetes, AWS, CI/CD pipelines and monitoring.",
        location="Pune, India",
        remote=False,
    ),
]


def active_jobs(session, source_key):
    return session.scalars(select(Job).where(Job.source == source_key, Job.is_active)).all()


def test_first_sync_inserts_and_embeds(session, embedder):
    stats = sync_postings(session, FakeSource(), CORPUS, embedder, NOW, EXPIRE)
    assert (stats.inserted, stats.updated, stats.unchanged, stats.embedded) == (4, 0, 0, 4)
    assert len(active_jobs(session, "greenhouse:acme")) == 4
    assert session.query(JobEmbedding).count() == 4


def test_resync_skips_unchanged_and_reembeds_changed(session, embedder):
    sync_postings(session, FakeSource(), CORPUS, embedder, NOW, EXPIRE)
    changed = [replace(CORPUS[0], description="Now also Go and gRPC."), *CORPUS[1:]]
    stats = sync_postings(session, FakeSource(), changed, embedder, NOW + timedelta(hours=6), EXPIRE)
    assert (stats.inserted, stats.updated, stats.unchanged, stats.embedded) == (0, 1, 3, 1)
    job = session.scalars(select(Job).where(Job.external_id == "1")).one()
    assert "gRPC" in job.description
    assert job.first_seen_at == NOW  # first_seen is preserved on update


def test_full_snapshot_retires_missing_jobs_and_revives_returning_ones(session, embedder):
    src = FakeSource()
    sync_postings(session, src, CORPUS, embedder, NOW, EXPIRE)
    stats = sync_postings(session, src, CORPUS[:3], embedder, NOW + timedelta(days=1), EXPIRE)
    assert stats.deactivated == 1
    assert {j.external_id for j in active_jobs(session, src.key)} == {"1", "2", "3"}
    sync_postings(session, src, CORPUS, embedder, NOW + timedelta(days=2), EXPIRE)
    assert len(active_jobs(session, src.key)) == 4


def test_partial_feed_retires_only_after_expiry(session, embedder):
    src = FakeSource(key="arbeitnow", full_snapshot=False)
    sync_postings(session, src, CORPUS, embedder, NOW, EXPIRE)
    stats = sync_postings(session, src, CORPUS[:1], embedder, NOW + timedelta(days=3), EXPIRE)
    assert stats.deactivated == 0
    stats = sync_postings(session, src, CORPUS[:1], embedder, NOW + timedelta(days=15), EXPIRE)
    assert stats.deactivated == 3


def test_empty_snapshot_does_not_wipe_a_source(session, embedder):
    src = FakeSource()
    sync_postings(session, src, CORPUS, embedder, NOW, EXPIRE)
    with pytest.raises(RuntimeError, match="refusing to retire"):
        sync_postings(session, src, [], embedder, NOW + timedelta(days=1), EXPIRE)


def test_duplicate_postings_in_one_fetch_are_collapsed(session, embedder):
    stats = sync_postings(session, FakeSource(), [CORPUS[0], CORPUS[0]], embedder, NOW, EXPIRE)
    assert stats.fetched == 1 and stats.inserted == 1


@pytest.fixture
def indexed(session, embedder):
    sync_postings(session, FakeSource(), CORPUS, embedder, NOW, EXPIRE)
    other = [
        posting(
            "9",
            "Python Developer",
            "Django and Python APIs.",
            company="Globex",
            location="Remote",
            remote=True,
        )
    ]
    sync_postings(session, FakeSource(key="remotive", full_snapshot=False), other, embedder, NOW, EXPIRE)
    return session


def test_search_ranks_relevant_jobs_first(indexed, embedder):
    hits = hybrid_search(indexed, embedder, "python backend intern")
    assert hits[0].title == "Backend Engineering Intern"
    assert "Senior Sales Manager" not in [h.title for h in hits[:3]]
    # The top hit was found by both retrievers.
    assert hits[0].keyword_rank is not None and hits[0].semantic_rank is not None


def test_search_or_semantics_for_chat_style_queries(indexed, embedder):
    # No job contains every one of these words, yet relevant jobs must still come back.
    hits = hybrid_search(indexed, embedder, "kubernetes terraform internship bangalore")
    assert hits and hits[0].title == "DevOps Engineer"


def test_search_filters(indexed, embedder):
    remote = hybrid_search(indexed, embedder, "python", SearchFilters(remote=True))
    assert hits_titles(remote) == {"Machine Learning Intern", "Python Developer"}

    india = hybrid_search(indexed, embedder, "engineer", SearchFilters(location="india"))
    assert all("India" in (h.location or "") for h in india)

    only_gh = hybrid_search(indexed, embedder, "python", SearchFilters(sources=("greenhouse",)))
    assert all(h.source.startswith("greenhouse:") for h in only_gh)

    no_remotive = hybrid_search(indexed, embedder, "python", SearchFilters(exclude_sources=("remotive",)))
    assert "Python Developer" not in hits_titles(no_remotive)


def test_search_skips_inactive_jobs(indexed, embedder):
    sync_postings(indexed, FakeSource(), CORPUS[1:], embedder, NOW + timedelta(days=1), EXPIRE)
    hits = hybrid_search(indexed, embedder, "backend engineering intern")
    assert "Backend Engineering Intern" not in hits_titles(hits)


def test_search_with_no_keyword_terms_still_uses_vectors(indexed, embedder):
    # "C++" yields an empty tsquery; the semantic side must still answer.
    hits = hybrid_search(indexed, embedder, "C++")
    assert all(h.keyword_rank is None for h in hits)


def test_search_rejects_empty_query(indexed, embedder):
    with pytest.raises(ValueError):
        hybrid_search(indexed, embedder, "   ")


def hits_titles(hits):
    return {h.title for h in hits}


def test_title_match_beats_boilerplate_mentions(session, embedder):
    """Regression: a description that opens with company boilerplate full of tech words used to
    outrank jobs whose titles actually match the query. Title-only retrievers fix this."""
    boilerplate = "We run Kubernetes, Terraform and DevOps tooling at scale. " * 8
    jobs = [
        posting("10", "Software Engineer 3", boilerplate + "You will write Java services."),
        posting("11", "DevOps Engineer", "Own our CI/CD and cloud infrastructure."),
        posting("12", "Site Reliability Engineer, Kubernetes", "Keep clusters healthy."),
        posting("13", "Cloud Infrastructure Engineer (Terraform)", "Infrastructure as code on AWS."),
    ]
    sync_postings(session, FakeSource(key="lever:acme"), jobs, embedder, NOW, EXPIRE)
    hits = hybrid_search(session, embedder, "devops engineer kubernetes terraform")
    assert hits[0].title == "DevOps Engineer"
    assert hits[0].keyword_rank == 1 and hits[0].semantic_rank == 1


def test_search_filters_by_field_and_level(session, embedder):
    jobs = [
        posting("f1", "Data Analyst Intern", "SQL and Python dashboards."),
        posting("f2", "Senior Data Engineer", "Spark and Python pipelines."),
        posting("f3", "Account Executive", "Sell Python training courses."),
    ]
    sync_postings(session, FakeSource(key="lever:fieldco"), jobs, embedder, NOW, EXPIRE)
    data = hybrid_search(session, embedder, "python", SearchFilters(fields=("data_ai",)))
    assert {h.title for h in data} == {"Data Analyst Intern", "Senior Data Engineer"}
    interns = hybrid_search(session, embedder, "python", SearchFilters(levels=("intern", "entry")))
    assert [h.title for h in interns] == ["Data Analyst Intern"] and interns[0].field == "data_ai"


def test_same_job_on_two_boards_is_stored_once(session, embedder):
    from datetime import UTC, datetime, timedelta

    from app.ingest.base import JobPosting
    from app.ingest.pipeline import sync_postings

    class Src:
        def __init__(self, key):
            self.key, self.full_snapshot = key, True

    def post(ext):
        return JobPosting(
            external_id=ext,
            title="Data Analyst Intern",
            company="Acme Corp.",
            location="Pune",
            url="http://x/" + ext,
            description="SQL and Excel.",
            posted_at=None,
            remote=False,
        )

    now = datetime.now(UTC)
    sync_postings(session, Src("board_a"), [post("a1")], embedder, now, timedelta(days=14))
    stats = sync_postings(session, Src("board_b"), [post("b1")], embedder, now, timedelta(days=14))
    assert stats.inserted == 0
    n = session.execute(text("SELECT count(*) FROM jobs WHERE company = 'Acme Corp.'")).scalar()
    assert n == 1


def test_india_filter(session, embedder):
    src = FakeSource()
    jobs = [
        posting("i1", "Backend Engineer", "Python APIs", location="Bengaluru, Karnataka"),
        posting("i2", "Backend Engineer II", "Python APIs", location="Remote - India"),
        posting("i3", "Backend Engineer III", "Python APIs", location="London, UK"),
        posting("i4", "Backend Engineer IV", "Python APIs", location="Indianapolis, IN"),
    ]
    sync_postings(session, src, jobs, embedder, datetime.now(UTC), timedelta(days=14))
    hits = hybrid_search(session, embedder, "backend engineer", SearchFilters(india=True), limit=10)
    assert {h.location for h in hits} == {"Bengaluru, Karnataka", "Remote - India"}
