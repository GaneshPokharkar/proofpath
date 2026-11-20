"""Last date to apply, and serving market reads from the database instead of recomputing."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select, text

from app.ingest.deadline import find_apply_by
from app.ingest.pipeline import sync_postings
from app.market.demand import cached_market_read
from app.models import IngestionRun, Job
from app.search import hybrid_search
from tests.test_market import FakeLLM
from tests.test_pipeline_and_search import EXPIRE, FakeSource, posting

POSTED = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Application Deadline: December 16, 2026\nAbout us", date(2026, 12, 16)),
        ("Application Deadline:\n\nJanuary 24, 2027", date(2027, 1, 24)),
        ("Last date to apply: 30/11/2026", date(2026, 11, 30)),
        ("Apply by 5th December 2026 to be considered.", date(2026, 12, 5)),
        ("Applications close 2026-12-20.", date(2026, 12, 20)),
        ("Application Deadline: ~January 2027", date(2027, 1, 31)),
        ("Founded in 2015. Start date: 1 Jan 2027.", None),  # dates without a deadline phrase
        ("Application deadline: rolling basis", None),
        ("Apply by 1 October 2026", None),  # before the posting date: not plausible
    ],
)
def test_find_apply_by(text, expected):
    assert find_apply_by(text, POSTED) == expected


def test_source_field_wins():
    assert find_apply_by("Apply by 1 Dec 2026", POSTED, field_value="2026-12-24T23:59:00Z") == date(
        2026, 12, 24
    )


def test_deadline_stored_and_expired_jobs_hidden_and_retired(session, embedder):
    today = datetime.now(UTC)
    soon = (today + timedelta(days=10)).strftime("%d/%m/%Y")
    past = (today - timedelta(days=2)).strftime("%d/%m/%Y")
    jobs = [
        posting(
            "d1",
            "Python Intern",
            f"Python internship. Last date to apply: {soon}",
            posted_at=today - timedelta(days=20),
        ),
        posting(
            "d2",
            "Python Developer Intern",
            f"Python role. Last date to apply: {past}",
            posted_at=today - timedelta(days=20),
        ),
    ]
    src = FakeSource(key="lever:deadlineco", full_snapshot=True)
    sync_postings(session, src, jobs, embedder, today - timedelta(days=5), EXPIRE)
    titles = {h.title: h for h in hybrid_search(session, embedder, "python intern")}
    assert "Python Intern" in titles and "Python Developer Intern" not in titles  # past deadline: hidden
    assert titles["Python Intern"].apply_by == (today + timedelta(days=10)).date()

    stats = sync_postings(session, src, jobs, embedder, today, EXPIRE)  # next refresh retires it
    assert stats.deactivated == 1
    assert session.scalar(select(Job.is_active).where(Job.external_id == "d2")) is False


def test_market_read_served_from_database_until_next_refresh(session, embedder):
    jobs = [
        posting("m1", "Backend Intern", "Python, FastAPI and Postgres."),
        posting("m2", "Backend Engineer", "Python and Kubernetes."),
    ]
    sync_postings(session, FakeSource(), jobs, embedder, datetime.now(UTC), EXPIRE)
    session.add(
        IngestionRun(
            source="x",
            started_at=datetime.now(UTC) - timedelta(hours=1),
            finished_at=datetime.now(UTC) - timedelta(hours=1),
            status="success",
        )
    )
    llm = FakeLLM()

    first, cached1 = cached_market_read(session, llm, embedder, "Backend  Engineer", sample=5)
    calls = llm.calls
    second, cached2 = cached_market_read(session, llm, embedder, "backend engineer", sample=5)
    assert (cached1, cached2) == (False, True)
    assert llm.calls == calls  # no new LLM calls, no re-search
    assert [s.skill for s in second.skills] == [s.skill for s in first.skills]
    assert session.scalar(text("SELECT hits FROM market_cache")) == 1

    # A newer successful refresh makes the saved answer stale, so it is rebuilt once.
    session.add(
        IngestionRun(
            source="x",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC) + timedelta(seconds=1),
            status="success",
        )
    )
    session.flush()
    _, cached3 = cached_market_read(session, llm, embedder, "backend engineer", sample=5)
    assert cached3 is False
