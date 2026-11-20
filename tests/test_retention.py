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
