"""Additive schema steps the app applies to itself on start (see app/api/main.py lifespan), so a
Vercel deploy never runs ahead of the Neon database. Alembic migrations reuse the same SQL."""

STEP_0009 = [
    # Two-layer chat memory. Keyed by a random id made in the visitor's browser: no name, email or
    # account. A rolling summary of older turns, the last few turns verbatim, and a profile the
    # agent fills with save_preference / shortlist. Deleted after MEMORY_DAYS without use.
    """CREATE TABLE IF NOT EXISTS agent_sessions (
        id TEXT PRIMARY KEY,
        summary TEXT NOT NULL DEFAULT '',
        recent JSONB NOT NULL DEFAULT '[]'::jsonb,
        profile JSONB NOT NULL DEFAULT '{}'::jsonb,
        turns INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )""",
    "CREATE INDEX IF NOT EXISTS agent_sessions_updated ON agent_sessions (updated_at)",
    # Anonymous counts per day and event ("search", "chat", "resume_match", ...). No ids.
    """CREATE TABLE IF NOT EXISTS usage_daily (
        day DATE NOT NULL,
        event TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (day, event)
    )""",
]


def ensure_schema(engine) -> list[str]:
    """Bring the live database up to migration 0009. Each step runs in its own transaction and
    only when it is missing, so an existing table never gets locked on a cold start."""
    from sqlalchemy import text

    from app.models import DEDUPE_SQL

    def has_column(conn, col: str) -> bool:
        return bool(
            conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns WHERE table_name = 'jobs' AND column_name = :c"
                ),
                {"c": col},
            ).first()
        )

    done = []
    steps = [
        (
            "last_served_at",
            lambda c: not has_column(c, "last_served_at"),
            ["ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_served_at TIMESTAMPTZ"],
        ),
        (
            "dedupe_key",
            lambda c: not has_column(c, "dedupe_key"),
            [
                f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS dedupe_key TEXT GENERATED ALWAYS AS ({DEDUPE_SQL}) STORED",
                "CREATE INDEX IF NOT EXISTS jobs_dedupe_key ON jobs (dedupe_key)",
            ],
        ),
        (
            "0009",
            lambda c: c.execute(
                text("SELECT to_regclass('agent_sessions') IS NULL OR to_regclass('usage_daily') IS NULL")
            ).scalar(),
            STEP_0009,
        ),
    ]
    for name, needed, sqls in steps:
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL lock_timeout = '5s'"))
            if needed(conn):
                for sql in sqls:
                    conn.execute(text(sql))
                done.append(name)
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE alembic_version SET version_num = '0009' WHERE version_num IN ('0006', '0007', '0008')"
            )
        )
    return done
