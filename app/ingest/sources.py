"""Connectors for public job APIs. Each has a pure `parse_*` function (unit-tested on
recorded payloads) and a small Source class that does the HTTP calls.

Terms of use worth knowing:
- Remotive: link back to the Remotive URL and name Remotive as the source; fetch at most
  4 times a day; its jobs must not be used to collect signups, so they are excluded from alerts.
- Greenhouse and Lever: public job-board APIs that companies publish for exactly this use.
"""

from dataclasses import dataclass

import httpx

from app.ingest.base import (
    JobPosting,
    clean,
    from_epoch,
    get_json,
    html_to_text,
    parse_iso,
)


def _date(value):
    from app.ingest.deadline import find_apply_by

    return find_apply_by("", field_value=value) if value else None


REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
LEVER_URL = "https://api.lever.co/v0/postings/{company}"


# --- Remotive -----------------------------------------------------------------------


def parse_remotive(payload: dict) -> list[JobPosting]:
    jobs = []
    for item in payload["jobs"]:
        tags = tuple(t for t in (item.get("tags") or []) if t)
        category = clean(item.get("category"))
        if category and category not in tags:
            tags = (category, *tags)
        jobs.append(
            JobPosting(
                external_id=str(item["id"]),
                title=item["title"].strip(),
                company=item["company_name"].strip(),
                url=item["url"],
                description=html_to_text(item.get("description")),
                location=clean(item.get("candidate_required_location")),
                remote=True,
                employment_type=clean(item.get("job_type")),
                tags=tags,
                salary_text=clean(item.get("salary")),
                posted_at=parse_iso(item.get("publication_date")),
            )
        )
    return jobs


@dataclass
class RemotiveSource:
    key: str = "remotive"
    full_snapshot: bool = False

    def fetch(self, client: httpx.Client) -> list[JobPosting]:
        return parse_remotive(get_json(client, REMOTIVE_URL))


# --- Arbeitnow ----------------------------------------------------------------------


def parse_arbeitnow(payload: dict) -> list[JobPosting]:
    jobs = []
    for item in payload["data"]:
        job_types = [t for t in (item.get("job_types") or []) if t]
        jobs.append(
            JobPosting(
                external_id=item["slug"],
                title=item["title"].strip(),
                company=item["company_name"].strip(),
                url=item["url"],
                description=html_to_text(item.get("description")),
                location=clean(item.get("location")),
                remote=bool(item.get("remote")),
                employment_type=", ".join(job_types) or None,
                tags=tuple(t for t in (item.get("tags") or []) if t),
                posted_at=from_epoch(item.get("created_at")),
            )
        )
    return jobs


@dataclass
class ArbeitnowSource:
    max_pages: int = 3
    key: str = "arbeitnow"
    full_snapshot: bool = False

    def fetch(self, client: httpx.Client) -> list[JobPosting]:
        jobs: list[JobPosting] = []
        url: str | None = ARBEITNOW_URL
        for _ in range(self.max_pages):
            if url is None:
                break
            payload = get_json(client, url)
            jobs.extend(parse_arbeitnow(payload))
            url = (payload.get("links") or {}).get("next")
        return jobs


# --- Greenhouse ---------------------------------------------------------------------


def parse_greenhouse(payload: dict, board: str) -> list[JobPosting]:
    jobs = []
    for item in payload["jobs"]:
        location = clean((item.get("location") or {}).get("name"))
        departments = tuple(d["name"] for d in (item.get("departments") or []) if d.get("name"))
        jobs.append(
            JobPosting(
                external_id=str(item["id"]),
                title=item["title"].strip(),
                company=clean(item.get("company_name")) or board,
                url=item["absolute_url"],
                description=html_to_text(item.get("content")),
                location=location,
                remote=("remote" in location.lower()) if location else None,
                tags=departments,
                posted_at=parse_iso(item.get("first_published") or item.get("updated_at")),
                apply_by=_date(item.get("application_deadline")),
            )
        )
    return jobs


@dataclass
class GreenhouseSource:
    board: str
    full_snapshot: bool = True

    @property
    def key(self) -> str:
        return f"greenhouse:{self.board}"

    def fetch(self, client: httpx.Client) -> list[JobPosting]:
        payload = get_json(client, GREENHOUSE_URL.format(board=self.board), {"content": "true"})
        return parse_greenhouse(payload, self.board)


# --- Lever --------------------------------------------------------------------------


def parse_lever(payload: list[dict], company: str) -> list[JobPosting]:
    jobs = []
    for item in payload:
        categories = item.get("categories") or {}
        workplace = clean(item.get("workplaceType"))
        tags = tuple(v for v in (categories.get("department"), categories.get("team")) if v)
        # Lever splits a posting into an intro, list sections ("What you will do") and a closing note.
        sections = [item.get("descriptionPlain") or ""]
        for section in item.get("lists") or []:
            sections.append(section.get("text", ""))
            sections.append(html_to_text(section.get("content")))
        sections.append(item.get("additionalPlain") or "")
        description = "\n\n".join(s.strip() for s in sections if s and s.strip())
        jobs.append(
            JobPosting(
                external_id=item["id"],
                title=item["text"].strip(),
                company=company,
                url=item["hostedUrl"],
                description=description,
                location=clean(categories.get("location")),
                remote=(workplace == "remote") if workplace else None,
                employment_type=clean(categories.get("commitment")),
                tags=tags,
                posted_at=from_epoch(item.get("createdAt"), millis=True),
            )
        )
    return jobs


@dataclass
class LeverSource:
    company: str
    display_name: str | None = None
    full_snapshot: bool = True

    @property
    def key(self) -> str:
        return f"lever:{self.company}"

    def fetch(self, client: httpx.Client) -> list[JobPosting]:
        payload = get_json(client, LEVER_URL.format(company=self.company), {"mode": "json"})
        return parse_lever(payload, self.display_name or self.company.title())
