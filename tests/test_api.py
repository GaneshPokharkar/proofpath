"""HTTP API, with the app's database session bound to the rolled-back test session."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.db import get_session
from app.ingest.pipeline import sync_postings
from tests.test_pipeline_and_search import CORPUS, FakeSource


@pytest.fixture
def client(session, embedder):
    sync_postings(session, FakeSource(), CORPUS, embedder, datetime.now(UTC), timedelta(days=14))
    app.state.embedder = embedder
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.embedder = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_endpoint(client):
    body = client.get("/search", params={"q": "devops kubernetes", "limit": 3}).json()
    assert body["count"] >= 1
    top = body["results"][0]
    assert top["title"] == "DevOps Engineer"
    assert top["source"] == "greenhouse:acme"
    assert body["took_ms"] >= 0


def test_search_validates_input(client):
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "limit": 500}).status_code == 422


def test_job_detail_and_404(client):
    job_id = client.get("/search", params={"q": "sales"}).json()["results"][0]["id"]
    detail = client.get(f"/jobs/{job_id}").json()
    assert detail["title"] == "Senior Sales Manager" and detail["is_active"] is True
    assert client.get("/jobs/999999999").status_code == 404


def test_stats(client):
    body = client.get("/stats").json()
    assert body["active_jobs"] == 4
    assert body["sources"][0]["source"] == "greenhouse:acme"


def test_market_needs_llm_key(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_api_key", None)
    r = client.get("/market", params={"role": "backend engineer"})
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]


def test_market_endpoint_with_fake_llm(client):
    from tests.test_market import FakeLLM

    app.state.llm = FakeLLM()
    try:
        body = client.get("/market", params={"role": "devops kubernetes", "sample": 5}).json()
    finally:
        del app.state.llm
    assert body["sample_size"] >= 1
    assert any(s["skill"] == "Kubernetes" for s in body["skills"])
