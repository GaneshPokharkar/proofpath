"""The master profile: everything true about the candidate, in one YAML file.

The tailor may choose, order and reword items from here for a job, but every claim on a generated
resume must trace back to this file. Skills the profile doesn't list are never added. Personal
profiles live in applier/profiles/ and are not committed (see .gitignore); example.yaml shows the
format.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Contact(BaseModel):
    name: str
    email: str
    phone: str  # with country code, e.g. +91 9876543210
    city: str
    state: str = ""
    country: str = "India"
    postal_code: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""


class Education(BaseModel):
    degree: str
    institution: str
    score: str
    year: str
    current: bool = False


class Project(BaseModel):
    id: str
    title: str
    tech: list[str]
    date: str
    links: dict[str, str] = Field(default_factory=dict)  # label -> url, e.g. {"Live": "...", "Code": "..."}
    facts: list[str] = Field(description="True statements about the work; the source for every bullet")
    bullets: list[str] = Field(description="Default bullets, already written")


class Experience(BaseModel):
    id: str
    org: str
    role: str
    dates: str
    facts: list[str]
    bullets: list[str]


class Answers(BaseModel):
    """Standard application questions, answered once by the candidate. The filler uses only these."""

    how_heard: str = "Social Media"
    notice_period: str = ""
    available_from: str = ""
    internship_duration: str = ""
    willing_to_relocate: str = ""
    work_authorization: str = ""
    expected_stipend: str = ""
    extra: dict[str, str] = Field(default_factory=dict)  # label fragment -> answer


class Profile(BaseModel):
    contact: Contact
    summary_facts: list[str]
    education: list[Education]
    projects: list[Project]
    experience: list[Experience] = Field(default_factory=list)
    skills: dict[str, list[str]]  # category -> skills the candidate really has
    coursework: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    answers: Answers = Field(default_factory=Answers)

    def all_skills(self) -> set[str]:
        return {s.lower() for group in self.skills.values() for s in group} | {
            t.lower() for p in self.projects for t in p.tech
        }

    def source_text(self) -> str:
        """Every true statement, for the fact-checker."""
        parts = list(self.summary_facts) + self.coursework + self.achievements
        for p in self.projects:
            parts += [p.title, ", ".join(p.tech), *p.facts, *p.bullets]
        for e in self.experience:
            parts += [e.org, e.role, *e.facts, *e.bullets]
        for g, items in self.skills.items():
            parts.append(f"{g}: {', '.join(items)}")
        for ed in self.education:
            parts.append(f"{ed.degree} {ed.institution} {ed.score} {ed.year}")
        return "\n".join(parts)


def load(path: str | Path) -> Profile:
    return Profile.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
