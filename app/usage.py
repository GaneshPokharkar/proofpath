"""Anonymous usage counts: how many searches, chats and resume checks happen per day. No ids."""

from sqlalchemy import text
from sqlalchemy.orm import Session


def count(session: Session, event: str) -> None:
    session.execute(
        text(
            "INSERT INTO usage_daily (day, event, count) VALUES (CURRENT_DATE, :e, 1) "
            "ON CONFLICT (day, event) DO UPDATE SET count = usage_daily.count + 1"
        ),
        {"e": event},
    )


def summary(session: Session, days: int = 30) -> dict:
    rows = session.execute(
        text(
            "SELECT event, sum(count) AS n FROM usage_daily WHERE day > CURRENT_DATE - :d "
            "GROUP BY event ORDER BY event"
        ),
        {"d": days},
    ).all()
    return {r.event: int(r.n) for r in rows}
