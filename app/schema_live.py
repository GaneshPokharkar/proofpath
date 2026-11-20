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
