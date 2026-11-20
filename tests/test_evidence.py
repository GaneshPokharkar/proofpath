"""GitHub evidence: detection rules, the API client (mocked), storage and readiness scoring."""

import base64
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.evidence.collect import analyze_student, readiness
from app.evidence.detect import detect, python_dependencies
from app.evidence.github import GitHubClient, GitHubError, RepoSnapshot
from app.ingest.pipeline import sync_postings
from app.market.demand import market_read
from tests.test_market import FakeLLM
from tests.test_pipeline_and_search import FakeSource, posting


def snap(**kw) -> RepoSnapshot:
    base = dict(
        full_name="u/r",
        url="https://github.com/u/r",
        description=None,
        is_fork=False,
        stars=0,
        pushed_at=None,
    )
    return RepoSnapshot(**{**base, **kw})


def by_skill(evidence):
    return {e.skill: e for e in evidence}


def test_languages_need_a_real_share_of_the_code():
    found = by_skill(detect(snap(languages={"Python": 90_000, "Shell": 500, "JavaScript": 1_500})))
    assert "Python" in found and "JavaScript" not in found  # 1.6% and under 2 KB: noise
    assert found["Python"].detail == "Python: 98% of code"


def test_dependencies_from_all_manifests():
    files = {
        "requirements.txt": "fastapi==0.115  # web\n-r base.txt\npsycopg[binary]>=3\nscikit_learn\n",
        "pyproject.toml": '[project]\ndependencies = ["torch>=2", "pgvector"]\n',
        "package.json": json.dumps(
            {"dependencies": {"react": "^19", "next": "15"}, "devDependencies": {"jest": "29"}}
        ),
    }
    found = by_skill(detect(snap(files=files)))
    for skill in ("FastAPI", "PostgreSQL", "scikit-learn", "PyTorch", "pgvector", "React", "Next.js", "Jest"):
        assert skill in found, skill
    assert found["FastAPI"].detail == "requirements.txt: fastapi==0.115"
    assert all(e.source == "dependency" for e in found.values())


def test_files_prove_tools_and_forks_prove_nothing():
    found = by_skill(detect(snap(files={"Dockerfile": "FROM python"}, workflows=["ci.yml"])))
    assert {"Docker", "GitHub Actions", "CI/CD"} <= found.keys()
    assert detect(snap(is_fork=True, languages={"Python": 10**6}, files={"Dockerfile": "x"})) == []


def test_broken_manifests_are_ignored():
    assert python_dependencies({"pyproject.toml": "not [valid toml"}) == []
    assert detect(snap(files={"package.json": "{broken"})) == []


# --- GitHub client against a fake API ------------------------------------------------------


def b64(text: str) -> dict:
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


FAKE_GITHUB = {
    "/users/varad/repos": [
        {
            "full_name": "varad/proofpath",
            "html_url": "https://github.com/varad/proofpath",
            "fork": False,
            "stargazers_count": 3,
            "pushed_at": "2026-11-19T10:00:00Z",
            "description": "AI project",
        },
        {
            "full_name": "varad/forked-lib",
            "html_url": "https://github.com/varad/forked-lib",
            "fork": True,
            "stargazers_count": 0,
            "pushed_at": "2026-10-01T10:00:00Z",
            "description": None,
        },
    ],
    "/repos/varad/proofpath/languages": {"Python": 120_000, "Dockerfile": 900},
    "/repos/varad/proofpath/readme": b64("# ProofPath\nHybrid search with pgvector."),
    "/repos/varad/proofpath/contents/": [
        {"name": "requirements.txt", "type": "file"},
        {"name": "Dockerfile", "type": "file"},
        {"name": ".github", "type": "dir"},
        {"name": "app", "type": "dir"},
    ],
    "/repos/varad/proofpath/contents/requirements.txt": b64("fastapi==0.165\npsycopg==3.3\npgvector\n"),
    "/repos/varad/proofpath/contents/Dockerfile": b64("FROM python:3.13-slim"),
    "/repos/varad/proofpath/contents/.github/workflows": [{"name": "ci.yml", "type": "file"}],
    "/repos/varad/forked-lib/languages": {"Python": 50_000},
    "/repos/varad/forked-lib/contents/": [{"name": "requirements.txt", "type": "file"}],
    "/repos/varad/forked-lib/contents/requirements.txt": b64("django\n"),
}


def fake_github(routes=FAKE_GITHUB, status=None):
    calls = []
    routes = {k.lower(): v for k, v in routes.items()}

    def handler(request: httpx.Request):
        calls.append(request.url.path)
        if status:
            return httpx.Response(status, headers={"x-ratelimit-remaining": "0"}, json={})
        path = request.url.path.lower()  # GitHub logins and repo names are case-insensitive
        if path in routes:
            return httpx.Response(200, json=routes[path])
        return httpx.Response(404, json={"message": "Not Found"})

    return GitHubClient(httpx.Client(transport=httpx.MockTransport(handler))), calls


def test_snapshot_reads_only_manifests_that_exist():
    gh, calls = fake_github()
    s = gh.snapshot(FAKE_GITHUB["/users/varad/repos"][0])
    assert set(s.files) == {"requirements.txt", "Dockerfile"}
    assert s.workflows == ["ci.yml"] and "pgvector" in s.readme
    assert "/repos/varad/proofpath/contents/package.json" not in calls  # absent: never fetched


def test_rate_limit_is_reported_clearly():
    gh, _ = fake_github(status=403)
    with pytest.raises(GitHubError, match="rate limit"):
        gh.list_repos("varad")


def test_analyze_student_stores_evidence_and_skips_forks(session, embedder):
    gh, _ = fake_github()
    summary = analyze_student(session, gh, embedder, "Varad")
    assert (summary.repos, summary.forks_skipped) == (2, 1)
    rows = session.execute(
        __import__("sqlalchemy").text(
            "SELECT sk.name, e.source FROM repo_evidence e JOIN skills sk ON sk.id = e.skill_id"
        )
    ).all()
    skills = {name for name, _ in rows}
    assert {"Python", "FastAPI", "PostgreSQL", "pgvector", "Docker", "CI/CD"} <= skills
    assert "Django" not in skills  # only in the fork
    # Re-running replaces evidence instead of duplicating it.
    again = analyze_student(session, gh, embedder, "varad")
    assert again.evidence == summary.evidence


def test_readiness_weights_skills_by_demand(session, embedder):
    now = datetime(2026, 11, 20, tzinfo=UTC)
    jobs = [
        posting("1", "Backend Intern", "Python, FastAPI and Postgres. Docker is a plus."),
        posting("2", "Backend Engineer", "Python and PostgreSQL on Kubernetes."),
        posting("3", "Python Developer", "Python and FastAPI. Kubernetes."),
    ]
    sync_postings(session, FakeSource(), jobs, embedder, now, timedelta(days=14))
    gh, _ = fake_github()
    analyze_student(session, gh, embedder, "varad")

    market = market_read(session, FakeLLM(), embedder, "python backend", sample=3, min_share=0.0)
    result = readiness(session, "varad", market)
    cov = {c.skill: c for c in result.skills}
    assert cov["Python"].proven and cov["FastAPI"].proven and cov["PostgreSQL"].proven
    assert not cov["Kubernetes"].proven and cov["Kubernetes"].repos == []
    assert cov["FastAPI"].details[0].startswith("varad/proofpath: requirements.txt")
    # Python 3/3 + FastAPI 2/3 + PostgreSQL 2/3 + Docker 1/3 proven; Kubernetes 2/3 missing.
    assert result.score == pytest.approx((3 + 2 + 2 + 1) / (3 + 2 + 2 + 1 + 2), abs=0.001)
