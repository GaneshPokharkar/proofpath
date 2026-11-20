"""Two-layer memory for the chat agent.

Layer 1, conversation: the last KEEP_TURNS turns verbatim plus a rolling summary of everything
older, rewritten by the LLM whenever turns fall out of the window.
Layer 2, profile: what the student told the agent to remember (skills, locations, remote, level,
field, salary floor, rejected roles) and their shortlist, written only through agent tools.

Both live in PostgreSQL under an anonymous id the browser generates; nothing names the person.
Rows are deleted after MEMORY_DAYS without use (app/retention.py) or at once on "forget me".
"""

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.llm import StructuredLLM

KEEP_TURNS = 6
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
PROFILE_KEYS = ("skills", "locations", "remote", "levels", "fields", "salary_floor", "rejected_roles")


@dataclass
class Memory:
    id: str
    summary: str = ""
    recent: list[dict] = field(default_factory=list)  # [{"user": ..., "agent": ...}]
    profile: dict = field(default_factory=dict)
    turns: int = 0

    @property
    def shortlist(self) -> list[int]:
        return list(self.profile.get("shortlist", []))

    def for_prompt(self) -> str:
        prof = {k: v for k, v in self.profile.items() if v not in (None, [], "")}
        parts = [f"Student profile (long-term memory): {json.dumps(prof) if prof else 'nothing saved yet'}"]
        if self.summary:
            parts.append(f"Summary of the earlier conversation: {self.summary}")
        return "\n".join(parts)


def load(session: Session, sid: str) -> Memory:
    row = session.execute(
        text("SELECT summary, recent, profile, turns FROM agent_sessions WHERE id = :i"), {"i": sid}
    ).first()
    if row is None:
        return Memory(sid)
    return Memory(sid, row.summary, list(row.recent), dict(row.profile), row.turns)


def save(session: Session, mem: Memory) -> None:
    session.execute(
        text(
            """INSERT INTO agent_sessions (id, summary, recent, profile, turns, updated_at)
               VALUES (:i, :s, CAST(:r AS jsonb), CAST(:p AS jsonb), :t, now())
               ON CONFLICT (id) DO UPDATE SET summary = :s, recent = CAST(:r AS jsonb),
                   profile = CAST(:p AS jsonb), turns = :t, updated_at = now()"""
        ),
        {
            "i": mem.id,
            "s": mem.summary,
            "r": json.dumps(mem.recent),
            "p": json.dumps(mem.profile),
            "t": mem.turns,
        },
    )


def forget(session: Session, sid: str) -> None:
    session.execute(text("DELETE FROM agent_sessions WHERE id = :i"), {"i": sid})


class Summary(BaseModel):
    summary: str = Field(description="At most 80 words: what the student wants and what was found or decided")


SUMMARY_PROMPT = """Update a running summary of a job-search chat between a student and an assistant.
Keep: the roles, places and constraints the student asked for, jobs they liked or rejected (with ids),
and decisions made. Drop greetings and anything already in the profile. At most 80 words."""


def add_turn(llm: StructuredLLM, mem: Memory, user: str, agent: str) -> None:
    """Append a turn; when the window overflows, fold the oldest turns into the rolling summary."""
    mem.recent.append({"user": user[:1000], "agent": agent[:1500]})
    mem.turns += 1
    if len(mem.recent) > KEEP_TURNS:
        old, mem.recent = mem.recent[:-KEEP_TURNS], mem.recent[-KEEP_TURNS:]
        transcript = "\n".join(f"Student: {t['user']}\nAssistant: {t['agent']}" for t in old)
        mem.summary = llm.extract(
            SUMMARY_PROMPT, f"Summary so far: {mem.summary or '(none)'}\n\nNew turns:\n{transcript}", Summary
        ).summary
