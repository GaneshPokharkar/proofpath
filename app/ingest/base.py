"""Shared types and helpers for job sources."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Protocol

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class JobPosting:
    """One job in the shape every source is normalised to."""

    external_id: str
    title: str
    company: str
    url: str
    description: str
    location: str | None = None
    remote: bool | None = None
    employment_type: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    salary_text: str | None = None
    posted_at: datetime | None = None
    apply_by: date | None = None  # last date to apply, when the source gives it as a field

    def content_hash(self) -> str:
        """Stable hash of everything that affects search; a change means re-embed."""
        parts = [
            self.title,
            self.company,
            self.location or "",
            str(self.remote),
            self.employment_type or "",
            "|".join(self.tags),
            self.salary_text or "",
            self.description,
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def embedding_text(self, max_description_chars: int = 1000) -> str:
        """Text the embedding model sees: the fields that say what the job is, title first."""
        lines = [self.title, self.company]
        if self.location:
            lines.append(self.location)
        if self.remote:
            lines.append("Remote")
        if self.tags:
            lines.append(", ".join(self.tags))
        lines.append(self.description[:max_description_chars])
        return "\n".join(lines)


class Source(Protocol):
    """A job source. `key` is stored in jobs.source, e.g. "greenhouse:groww"."""

    key: str
    # True when every fetch returns the source's complete list of open jobs, so a job
    # missing from a fetch has closed. False for paginated or partial feeds.
    full_snapshot: bool

    def fetch(self, client: httpx.Client) -> list[JobPosting]: ...


_TAG = re.compile(r"<[a-zA-Z/][^>]*>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def html_to_text(raw: str | None) -> str:
    """HTML (possibly entity-escaped, as Greenhouse and Arbeitnow send it) to readable plain text."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text("\n")
    # Escaped HTML ("&lt;p&gt;", as Greenhouse and Arbeitnow send it) comes out of the first
    # parse as literal tags; a second parse turns those into text.
    if _TAG.search(text):
        text = BeautifulSoup(text, "html.parser").get_text("\n")
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def clean(value: str | None) -> str | None:
    """Strip whitespace; empty strings become None."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def from_epoch(value: int | float | None, *, millis: bool = False) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000 if millis else value, tz=UTC)


def get_json(client: httpx.Client, url: str, params: dict | None = None):
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()
