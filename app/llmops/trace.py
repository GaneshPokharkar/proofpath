"""Every LLM call leaves one row in llm_calls: what it was for, which provider and model answered,
how long it took, how many tokens it used, whether it failed or fell back, or was a cache hit.

No prompt or answer text is stored (they can hold a student's resume); only these numbers.
The sink is swappable so tests can collect records in memory.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from sqlalchemy import text

log = logging.getLogger(__name__)


@dataclass
class CallRecord:
    task: str  # e.g. "chat", "JobSkills", "ResumeProfile", "market_report"
    provider: str  # "groq", "gemini", "gemini-byok", "cache", "semantic-cache"
    model: str
    ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    ok: bool = True
    error: str = ""
    fallback: bool = False  # this provider answered because an earlier one failed
    extra: dict = field(default_factory=dict)


def _db_sink(rec: CallRecord) -> None:
    from app.db import get_engine

    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO llm_calls (task, provider, model, ms, tokens_in, tokens_out, ok, error, fallback)
                       VALUES (:task, :provider, :model, :ms, :tokens_in, :tokens_out, :ok, :error, :fallback)"""
                ),
                {k: v for k, v in asdict(rec).items() if k != "extra"},
            )
    except Exception as exc:  # noqa: BLE001 - tracing must never break a student's request
        log.warning("could not store LLM trace: %s", exc)


sink: Callable[[CallRecord], None] = _db_sink


def record(rec: CallRecord) -> None:
    sink(rec)


@contextmanager
def timed(rec: CallRecord) -> Iterator[CallRecord]:
    """Fill in ms / ok / error around a call and record it, whatever happens."""
    start = time.perf_counter()
    try:
        yield rec
    except Exception as exc:
        rec.ok, rec.error = False, f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        rec.ms = int((time.perf_counter() - start) * 1000)
        record(rec)


def summary(session, days: int = 7) -> dict:
    """Per task and provider: calls, errors, fallbacks, p50/p95 latency and tokens."""
    rows = (
        session.execute(
            text(
                """SELECT task, provider, model, count(*) AS calls,
                      count(*) FILTER (WHERE NOT ok) AS errors,
                      count(*) FILTER (WHERE fallback) AS fallbacks,
                      percentile_disc(0.5) WITHIN GROUP (ORDER BY ms) AS p50_ms,
                      percentile_disc(0.95) WITHIN GROUP (ORDER BY ms) AS p95_ms,
                      sum(tokens_in) AS tokens_in, sum(tokens_out) AS tokens_out
               FROM llm_calls WHERE at > now() - make_interval(days => :d)
               GROUP BY task, provider, model ORDER BY calls DESC"""
            ),
            {"d": days},
        )
        .mappings()
        .all()
    )
    by = [
        dict(r) | {"tokens_in": int(r["tokens_in"] or 0), "tokens_out": int(r["tokens_out"] or 0)}
        for r in rows
    ]
    llm = [r for r in by if r["provider"] not in ("cache", "semantic-cache")]
    cached = sum(r["calls"] for r in by if r["provider"] in ("cache", "semantic-cache"))
    calls = sum(r["calls"] for r in llm)
    return {
        "days": days,
        "llm_calls": calls,
        "errors": sum(r["errors"] for r in llm),
        "fallbacks": sum(r["fallbacks"] for r in llm),
        "tokens": sum(r["tokens_in"] + r["tokens_out"] for r in llm),
        "cache_hits": cached,
        "cache_hit_rate": round(cached / (cached + calls), 3) if cached + calls else None,
        "by_task": by,
    }
