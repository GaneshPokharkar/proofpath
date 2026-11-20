"""Test setup: a real PostgreSQL + pgvector database, migrated with Alembic once per session.

Set JOBPILOT_TEST_DATABASE_URL to an empty database (CI provides one as a service container).
Every test runs inside a transaction that is rolled back afterwards.
"""

import json
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.embeddings import HashingEmbedder

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

TEST_DB_URL = os.environ.get("JOBPILOT_TEST_DATABASE_URL")
if not TEST_DB_URL:
    raise RuntimeError(
        "JOBPILOT_TEST_DATABASE_URL is not set. Point it at an empty PostgreSQL database "
        "with pgvector, e.g. postgresql+psycopg://postgres:postgres@localhost:5432/jobpilot_test"
    )
# The app reads JOBPILOT_DATABASE_URL; tests must never touch any other database.
os.environ["JOBPILOT_DATABASE_URL"] = TEST_DB_URL


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(TEST_DB_URL)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    """A session whose work is rolled back at the end, including commits made by code under test."""
    connection = engine.connect()
    outer = connection.begin()
    sess = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield sess
    sess.close()
    outer.rollback()
    connection.close()


@pytest.fixture
def embedder():
    return HashingEmbedder()
