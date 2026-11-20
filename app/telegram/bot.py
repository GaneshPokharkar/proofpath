"""Telegram bot logic, separate from HTTP so it is easy to test.

handle_update() takes one Telegram update and returns the reply text; send_alerts() builds the
daily alert messages. Sending is done by TelegramClient, which the tests replace.
"""

import html
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.models import Job
from app.models_telegram import TelegramChat
from app.search import SearchFilters, hybrid_search

API = "https://api.telegram.org/bot{token}/{method}"

SITE = "https://proofpath-ganeshpokharkars-projects.vercel.app"

HELP = (
    "Hi! I'm <b>ProofPath</b>. I search live job postings in India for you.\n\n"
    "<b>/jobs</b> <i>what you want</i> - e.g. <code>/jobs python intern bangalore</code>\n"
    "<b>/skills</b> <i>a role</i> - which skills employers ask for, e.g. <code>/skills data analyst</code>\n"
    "<b>/alerts</b> <i>a search</i> - I'll message you new matching jobs every day\n"
    "<b>/alerts off</b> - stop daily alerts\n"
    "<b>/help</b> - this message\n\n"
    "Free for everyone. I keep only your chat number, and only while daily alerts are on."
)


class TelegramClient:
    def __init__(self, token: str, client: httpx.Client | None = None):
        self._token = token
        self._client = client or httpx.Client(timeout=20)

    def call(self, method: str, **params) -> dict:
        r = self._client.post(API.format(token=self._token, method=method), json=params)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {data.get('description')}")
        return data["result"]

    def send(self, chat_id: int, text: str) -> None:
        self.call(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            link_preview_options={"is_disabled": True},
        )


@dataclass
class Deps:
    session: Session
    embedder_factory: Callable[[], Embedder]  # returns an Embedder; lazy so /help never loads the model
    admins: set[str]
    refresh: Callable[..., object]  # (force: bool) -> RefreshResult
    market: Callable[[str], object] | None  # (role: str) -> MarketRead, None when no LLM key is set


def _esc(s: str | None) -> str:
    return html.escape(s or "")


def format_jobs(hits, limit: int = 5) -> str:
    lines = []
    for i, h in enumerate(hits[:limit], 1):
        where = h.location or ("Remote" if h.remote else "")
        lines.append(
            f'{i}. <a href="{_esc(h.url)}">{_esc(h.title)}</a>\n   {_esc(h.company)} · {_esc(where)}'
        )
    return "\n".join(lines)


def handle_update(update: dict, deps: Deps) -> tuple[int, str] | None:
    """Returns (chat_id, reply) for a message, or None for updates the bot ignores."""
    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return None
    chat_id = msg["chat"]["id"]
    username = (msg.get("from") or {}).get("username") or ""
    text = msg["text"].strip()
    command, _, arg = text.partition(" ")
    command = command.split("@")[0].lower()  # "/jobs@ProofPathBot" in groups
    arg = arg.strip()

    deps.session.execute(
        pg_insert(TelegramChat)
        .values(chat_id=chat_id)  # only the chat number; no names are stored
        .on_conflict_do_nothing(index_elements=[TelegramChat.chat_id])
    )

    if command in ("/start", "/help"):
        return chat_id, HELP

    if command == "/jobs" or not command.startswith("/"):
        query = arg if command == "/jobs" else text
        if not query:
            return chat_id, "Tell me what you're looking for, e.g. <code>/jobs react developer remote</code>"
        hits = hybrid_search(deps.session, deps.embedder_factory(), query, SearchFilters(india=True), limit=5)
        if not hits:
            return chat_id, f"No open jobs match <b>{_esc(query)}</b> yet. Try fewer or broader words."
        return chat_id, f"Top jobs for <b>{_esc(query)}</b>:\n\n{format_jobs(hits)}"

    if command == "/skills":
        if not arg:
            return chat_id, "Which role? e.g. <code>/skills machine learning engineer</code>"
        from app.market.demand import NeedsKey

        try:
            read = deps.market(arg)
        except NeedsKey:
            return chat_id, (
                f"Nobody has checked <b>{_esc(arg)}</b> this week yet. Open {SITE}, add your free "
                "Gemini key and look it up once; after that I'll answer it instantly for everyone."
            )
        if not read.skills:
            return chat_id, f"I couldn't find enough postings for <b>{_esc(arg)}</b>."
        rows = [
            f"{i}. <b>{_esc(s.skill)}</b> - {round(s.share * 100)}% of postings"
            for i, s in enumerate(read.skills[:10], 1)
        ]
        return chat_id, (
            f"What employers ask for in <b>{_esc(arg)}</b> "
            f"(from {read.sample_size} live postings):\n\n" + "\n".join(rows)
        )

    if command == "/alerts":
        chat = deps.session.get(TelegramChat, chat_id)
        if arg.lower() in ("off", "stop"):
            chat.alerts_on = False
            return chat_id, "Daily alerts are off. Send /alerts <i>a search</i> to turn them back on."
        if not arg:
            state = f"on for <b>{_esc(chat.alert_query)}</b>" if chat.alerts_on else "off"
            return chat_id, f"Daily alerts are {state}. Example: <code>/alerts python intern india</code>"
        chat.alert_query, chat.alerts_on, chat.last_alert_at = arg, True, datetime.now(UTC)
        return chat_id, f"Done. Every day I'll send you new jobs matching <b>{_esc(arg)}</b>."

    if command == "/refresh":
        if username.lower() not in deps.admins:
            return chat_id, "Only the ProofPath admin can refresh jobs. They update automatically every day."
        result = deps.refresh(force=arg == "force")
        return chat_id, f"Refresh: {_esc(result.message)}"

    return chat_id, "I don't know that command. Send /help to see what I can do."


def build_alerts(session: Session, embedder: Embedder, now: datetime | None = None) -> list[tuple[int, str]]:
    """New jobs (first seen since each chat's last alert) matching each chat's saved search.
    Remotive jobs are excluded: its terms forbid using its listings to collect signups."""
    now = now or datetime.now(UTC)
    messages = []
    for chat in session.scalars(select(TelegramChat).where(TelegramChat.alerts_on)):
        hits = hybrid_search(
            session,
            embedder,
            chat.alert_query,
            SearchFilters(exclude_sources=("remotive",), india=True),
            limit=30,
        )
        since = chat.last_alert_at or now
        fresh_ids = set(
            session.scalars(
                select(Job.id).where(Job.id.in_([h.job_id for h in hits]), Job.first_seen_at > since)
            )
        )
        fresh = [h for h in hits if h.job_id in fresh_ids]
        chat.last_alert_at = now
        if fresh:
            messages.append(
                (chat.chat_id, f"New jobs for <b>{_esc(chat.alert_query)}</b>:\n\n{format_jobs(fresh)}")
            )
    return messages
