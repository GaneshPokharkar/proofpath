from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.retention import purge


def test_purge_removes_everything_older_than_a_day(session):
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    t = {"t": old, "n": now}
    session.execute(
        text("INSERT INTO students (github_login, created_at) VALUES ('olduser', :t), ('newuser', :n)"), t
    )
    session.execute(text("INSERT INTO telegram_chats (chat_id, created_at) VALUES (1, :t), (2, :n)"), t)
    session.commit()
    out = purge(session, now)
    assert out["github_checks"] == 1 and out["telegram_chats"] == 1
    assert session.execute(text("SELECT github_login FROM students")).scalars().all() == ["newuser"]


def test_no_fresh_refresh_keeps_jobs(session):
    out = purge(session)
    assert "stale_jobs" not in out


def _job(session, ext, seen, served, active=True):
    session.execute(
        text("""INSERT INTO jobs (source, external_id, title, company, url, description,
        content_hash, first_seen_at, last_seen_at, last_served_at, is_active)
        VALUES ('t', :e, 'Dev', 'Co', 'http://x', 'd', :h, :s, :s, :v, :a)"""),
        {"e": ext, "h": ext, "s": seen, "v": served, "a": active},
    )


def test_jobs_are_never_deleted(session):
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    session.execute(
        text("""INSERT INTO ingestion_runs (source, started_at, finished_at, status)
        VALUES ('t', :n, :n, 'success')"""),
        {"n": now},
    )
    _job(session, "searched", old, now)
    _job(session, "searched_closed", old, now, active=False)
    _job(session, "ignored", old, None)
    _job(session, "today", now, None)
    session.commit()
    purge(session, now)
    left = set(session.execute(text("SELECT external_id FROM jobs WHERE source='t'")).scalars())
    assert left == {"searched", "searched_closed", "ignored", "today"}
