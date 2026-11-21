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


STEP_0010 = [
    # One row per LLM call (or cache hit): numbers only, never prompt or answer text.
    """CREATE TABLE IF NOT EXISTS llm_calls (
        id BIGSERIAL PRIMARY KEY,
        at TIMESTAMPTZ NOT NULL DEFAULT now(),
        task TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        ms INTEGER NOT NULL DEFAULT 0,
        tokens_in INTEGER NOT NULL DEFAULT 0,
        tokens_out INTEGER NOT NULL DEFAULT 0,
        ok BOOLEAN NOT NULL DEFAULT true,
        error TEXT NOT NULL DEFAULT '',
        fallback BOOLEAN NOT NULL DEFAULT false
    )""",
    "CREATE INDEX IF NOT EXISTS llm_calls_at ON llm_calls (at)",
    # Semantic cache: the role's embedding and the filters it was built for (scope).
    "ALTER TABLE market_cache ADD COLUMN IF NOT EXISTS scope TEXT",
    "ALTER TABLE market_cache ADD COLUMN IF NOT EXISTS role_vec vector(384)",
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
        (
            "0010",
            lambda c: c.execute(
                text(
                    "SELECT to_regclass('llm_calls') IS NULL OR NOT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'market_cache' AND column_name = 'role_vec')"
                )
            ).scalar(),
            STEP_0010,
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
                "UPDATE alembic_version SET version_num = '0010' WHERE version_num IN ('0006', '0007', '0008', '0009')"
            )
        )
    return done
