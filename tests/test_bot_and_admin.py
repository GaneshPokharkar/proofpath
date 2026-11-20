"""Telegram bot logic, refresh safeguards, admin/cron auth and the web pages."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.api.main import _webhook_secret, app
from app.config import get_settings
from app.db import get_session
from app.ingest.pipeline import sync_postings
from app.models_telegram import TelegramChat
from app.telegram.bot import Deps, build_alerts, handle_update
from tests.test_pipeline_and_search import CORPUS, FakeSource, posting

NOW = datetime(2026, 11, 20, 6, 0, tzinfo=UTC)


def msg(text, username="student1", chat_id=111):
    return {"message": {"chat": {"id": chat_id}, "from": {"username": username}, "text": text}}


@pytest.fixture
def deps(session, embedder):
    sync_postings(session, FakeSource(), CORPUS, embedder, NOW, timedelta(days=14))
    calls = []

    def refresh(force):
        calls.append(force)
        return type("R", (), {"message": "2 new, 0 updated, 0 closed across 18 sources"})()

    d = Deps(
        session=session, embedder_factory=lambda: embedder, admins={"varad"}, refresh=refresh, market=None
    )
    d.calls = calls
    return d


def test_help_and_unknown(deps):
    assert "/jobs" in handle_update(msg("/start"), deps)[1]
    assert "don't know" in handle_update(msg("/dance"), deps)[1]
    assert handle_update({"edited_channel_post": {}}, deps) is None


def test_jobs_command_and_plain_text_search(deps):
    chat, reply = handle_update(msg("/jobs devops kubernetes"), deps)
    assert chat == 111 and "DevOps Engineer" in reply and "https://jobs.example.com/4" in reply
    assert "Backend Engineering Intern" in handle_update(msg("python backend intern"), deps)[1]
    assert "Tell me what" in handle_update(msg("/jobs"), deps)[1]


def test_bot_escapes_html_in_user_text(deps):
    reply = handle_update(msg("/alerts <b>x</b>"), deps)[1]
    assert "&lt;b&gt;x&lt;/b&gt;" in reply


def test_skills_without_saved_report_points_to_site(deps):
    from app.market.demand import NeedsKey

    def no_report(role):
        raise NeedsKey(role)

    deps.market = no_report
    reply = handle_update(msg("/skills data analyst"), deps)[1]
    assert "Nobody has checked" in reply and "Gemini key" in reply


def test_refresh_is_admin_only(deps):
    assert "Only the ProofPath admin" in handle_update(msg("/refresh", username="stranger"), deps)[1]
    assert deps.calls == []
    assert "2 new" in handle_update(msg("/refresh force", username="Varad"), deps)[1]
    assert deps.calls == [True]


def test_alerts_send_only_new_matching_jobs(deps, session, embedder):
    handle_update(msg("/alerts python intern"), deps)
    chat = session.get(TelegramChat, 111)
    assert chat.alerts_on and chat.alert_query == "python intern"
    chat.last_alert_at = NOW  # jobs above were first seen at NOW, i.e. not new

    assert build_alerts(session, embedder, now=NOW + timedelta(days=1)) == []
    later = NOW + timedelta(days=2)
    sync_postings(
        session,
        FakeSource(),
        [*CORPUS, posting("n1", "Python Intern", "Python and FastAPI.", location="Pune, India")],
        embedder,
        later,
        timedelta(days=14),
    )
    remotive = [
        posting("r1", "Python Intern (Remotive)", "Python internship.", remote=True, location="India")
    ]
    sync_postings(
        session,
        FakeSource(key="remotive", full_snapshot=False),
        remotive,
        embedder,
        later,
        timedelta(days=14),
    )
    alerts = build_alerts(session, embedder, now=later + timedelta(hours=1))
    assert len(alerts) == 1 and alerts[0][0] == 111
    assert "Python Intern" in alerts[0][1] and "Remotive" not in alerts[0][1]
    assert build_alerts(session, embedder, now=later + timedelta(hours=2)) == []  # already sent

    handle_update(msg("/alerts off"), deps)
    assert session.get(TelegramChat, 111).alerts_on is False


# --- HTTP: admin, cron, webhook, pages -------------------------------------------------------


@pytest.fixture
def client(session, embedder, monkeypatch):
    settings = get_settings()
    for name in ("admin_secret", "cron_secret", "telegram_bot_token"):
        monkeypatch.setattr(settings, name, None)
    app.state.embedder = embedder
    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        "app.api.main._refresh",
        lambda request, force: type(
            "R", (), {"__dict__": {"status": "done", "message": "ok", "force": force}}
        )(),
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.embedder = None


def test_pages_render(client):
    home = client.get("/")
    assert home.status_code == 200 and "Prove you're ready" in home.text
    assert "Admin password" in client.get("/admin").text


def test_admin_refresh_placeholder_then_password(client, monkeypatch):
    r = client.post("/admin/refresh")
    assert r.status_code == 503 and "ADMIN_SECRET" in r.json()["detail"]
    monkeypatch.setattr(get_settings(), "admin_secret", "s3cret")
    assert client.post("/admin/refresh", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.post("/admin/refresh?force=true", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.json()["force"] is True


def test_cron_auth(client, monkeypatch):
    assert client.get("/cron/daily").status_code == 401
    r = client.get("/cron/daily", headers={"User-Agent": "vercel-cron/1.0"})
    assert r.status_code == 200 and r.json()["alerts_sent"] == 0  # no bot token: no alerts
    monkeypatch.setattr(get_settings(), "cron_secret", "c")
    assert client.get("/cron/daily", headers={"User-Agent": "vercel-cron/1.0"}).status_code == 401
    assert client.get("/cron/daily", headers={"Authorization": "Bearer c"}).status_code == 200


def test_status_reveals_no_secrets(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "admin_secret", "s3cret")
    body = client.get("/admin/status").json()
    assert body["configured"]["admin_password"] is True and "s3cret" not in str(body)


def test_webhook_rejects_requests_not_from_telegram(client, monkeypatch):
    assert client.post("/telegram/webhook", json={}).status_code == 503
    monkeypatch.setattr(get_settings(), "telegram_bot_token", "123:abc")
    assert client.post("/telegram/webhook", json={}).status_code == 401
    sent = []
    monkeypatch.setattr("app.telegram.bot.TelegramClient.send", lambda self, chat, text: sent.append(chat))
    monkeypatch.setattr("app.api.main.session_scope", None, raising=False)
    hdr = {"X-Telegram-Bot-Api-Secret-Token": _webhook_secret("123:abc")}
    r = client.post("/telegram/webhook", json={"update_id": 1}, headers=hdr)
    assert r.status_code == 200 and sent == []  # an update with no message is ignored


def test_refresh_cooldown_and_lock(session, embedder):
    from contextlib import nullcontext

    from sqlalchemy import text

    from app.db import get_engine
    from app.models import IngestionRun
    from app.refresh import LOCK_KEY, run_refresh

    factory = lambda: nullcontext(session)  # noqa: E731
    session.add(
        IngestionRun(
            source="x", started_at=datetime.now(UTC), finished_at=datetime.now(UTC), status="success"
        )
    )
    session.flush()
    assert run_refresh(embedder, sources=[], session_factory=factory).status == "skipped_recent"

    with get_engine().connect() as other:  # someone else holds the lock
        other.execute(text("SELECT pg_advisory_lock(:k)"), {"k": LOCK_KEY})
        assert (
            run_refresh(embedder, force=True, sources=[], session_factory=factory).status == "skipped_running"
        )
        other.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_KEY})

    done = run_refresh(embedder, force=True, sources=[], session_factory=factory)
    assert done.status == "done" and done.message.startswith("0 new")
