"""Load evals/jobs.json into the database the evals run against (idempotent)."""

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from app.ingest.base import JobPosting, parse_iso
from app.ingest.pipeline import sync_postings

FIXTURE = Path(__file__).with_name("jobs.json")


@dataclass
class FixtureSource:
    key: str
    postings: list[JobPosting] = field(default_factory=list)
    full_snapshot: bool = True

    def fetch(self, client):
        return self.postings


def load_postings() -> dict[str, list[JobPosting]]:
    by_source: dict[str, list[JobPosting]] = {}
    for j in json.loads(FIXTURE.read_text()):
        by_source.setdefault(j["source"], []).append(
            JobPosting(
                external_id=j["external_id"],
                title=j["title"],
                company=j["company"],
                url=j["url"],
                description=j["description"],
                location=j["location"],
                remote=j["remote"],
                posted_at=parse_iso(j["posted_at"]),
                apply_by=date.fromisoformat(j["apply_by"]) if j.get("apply_by") else None,
            )
        )
    return by_source


def seed(session, embedder) -> int:
    n = 0
    for key, postings in load_postings().items():
        sync_postings(
            session, FixtureSource(key, postings), postings, embedder, datetime.now(UTC), timedelta(days=3650)
        )
        n += len(postings)
    session.commit()
    return n
