"""Parsers against real (trimmed) API responses recorded on 2026-11-20."""

from datetime import datetime

import pytest

from app.ingest.base import JobPosting, html_to_text
from app.ingest.registry import load_sources
from app.ingest.sources import parse_arbeitnow, parse_greenhouse, parse_lever, parse_remotive
from tests.conftest import load_fixture


def _check_common(jobs: list[JobPosting], expected: int):
    assert len(jobs) == expected
    for job in jobs:
        assert job.external_id and job.title and job.company
        assert job.url.startswith("https://")
        assert len(job.description) > 100
        # Parsed descriptions must be plain text, never raw or escaped HTML.
        assert "<p" not in job.description and "&lt;" not in job.description
        assert job.posted_at is None or (isinstance(job.posted_at, datetime) and job.posted_at.tzinfo)


def test_parse_remotive():
    jobs = parse_remotive(load_fixture("remotive.json"))
    _check_common(jobs, 3)
    assert all(j.remote is True for j in jobs)
    assert jobs[0].title == "Tech Lead Full-Stack Rails Engineer"
    assert jobs[0].salary_text == "$170k - $200k"
    assert "Software Development" in jobs[0].tags


def test_parse_arbeitnow():
    jobs = parse_arbeitnow(load_fixture("arbeitnow.json"))
    _check_common(jobs, 3)
    assert jobs[0].company == "Xometryeurope"
    assert jobs[0].location == "Munich"
    assert jobs[0].remote is False


def test_parse_greenhouse_unescapes_html():
    jobs = parse_greenhouse(load_fixture("greenhouse.json"), "stripe")
    _check_common(jobs, 3)
    assert jobs[0].company == "Stripe"
    assert jobs[0].external_id == "8172510"
    assert "Who we are" in jobs[0].description


def test_parse_lever_joins_sections():
    jobs = parse_lever(load_fixture("lever.json"), "Meesho")
    _check_common(jobs, 3)
    first = jobs[0]
    assert first.company == "Meesho"
    assert first.employment_type == "Full Time Employee"
    assert first.remote is False  # workplaceType "onsite"
    assert "What you will do" in first.description


def test_html_to_text_handles_escaped_and_plain():
    assert html_to_text("&lt;p&gt;Hello &amp;amp; welcome&lt;/p&gt;") == "Hello & welcome"
    assert html_to_text("<ul><li>One</li><li>Two</li></ul>") == "One\nTwo"
    assert html_to_text(None) == ""


def test_content_hash_changes_only_with_content():
    a = JobPosting("1", "Backend Intern", "Acme", "https://x", "Build APIs")
    assert (
        a.content_hash()
        == JobPosting("1", "Backend Intern", "Acme", "https://x", "Build APIs").content_hash()
    )
    assert (
        a.content_hash()
        != JobPosting("1", "Backend Intern", "Acme", "https://x", "Build APIs!").content_hash()
    )


def test_registry_loads_repo_config():
    sources = load_sources("sources.yaml")
    keys = [s.key for s in sources]
    assert "remotive" in keys and "greenhouse:groww" in keys and "lever:meesho" in keys
    assert len(keys) == len(set(keys))


def test_registry_rejects_unknown_source(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text("linkedin: [foo]\n")
    with pytest.raises(ValueError, match="unknown source types"):
        load_sources(bad)
