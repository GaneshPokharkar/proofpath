"""Runtime settings, read from environment variables (prefix JOBPILOT_) or a .env file."""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="JOBPILOT_", extra="ignore")

    # Required: there is deliberately no default, so a missing database URL fails loudly at startup.
    # DATABASE_URL is what Vercel's Neon integration sets; JOBPILOT_DATABASE_URL wins if both exist.
    database_url: str = Field(validation_alias=AliasChoices("JOBPILOT_DATABASE_URL", "DATABASE_URL"))

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3(cls, url: str) -> str:
        # Hosted Postgres hands out postgres:// or postgresql:// URLs; SQLAlchemy needs the driver named.
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix) :]
        return url

    # Must produce vectors of app.models.EMBEDDING_DIM dimensions (384 for bge-small).
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    sources_file: str = "sources.yaml"
    # A posting not seen for this many days is marked inactive (for sources that are not full snapshots).
    expire_after_days: int = 14
    http_timeout_s: float = 30.0
    user_agent: str = "JobPilot/0.1 (+https://github.com/GaneshPokharkar/jobpilot)"

    # LLM (skill extraction and, later, the agent). Optional so ingestion and search run without it;
    # endpoints that need it return 503 until it is set. Gemini has a free tier; Anthropic is paid.
    llm_provider: str = "gemini"
    llm_api_key: str | None = None
    llm_model: str = "gemini-flash-lite-latest"

    # Secrets below are optional placeholders: each feature reports "not set up yet" until its
    # value exists. They read the plain names used in the Vercel dashboard.
    # ADMIN_SECRET guards /admin (refresh button); Vercel Cron sends CRON_SECRET automatically.
    admin_secret: str | None = Field(
        None, validation_alias=AliasChoices("JOBPILOT_ADMIN_SECRET", "ADMIN_SECRET")
    )
    cron_secret: str | None = Field(
        None, validation_alias=AliasChoices("JOBPILOT_CRON_SECRET", "CRON_SECRET")
    )
    telegram_bot_token: str | None = Field(
        None, validation_alias=AliasChoices("JOBPILOT_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
    )
    # Comma-separated Telegram usernames (without @) allowed to run /refresh.
    telegram_admins: str = Field(
        "", validation_alias=AliasChoices("JOBPILOT_TELEGRAM_ADMINS", "TELEGRAM_ADMINS")
    )
    github_token: str | None = Field(
        None, validation_alias=AliasChoices("JOBPILOT_GITHUB_TOKEN", "GITHUB_TOKEN")
    )
    # A refresh within this many hours of the last successful one is skipped unless forced.
    latex_compile_url: str = "https://latex.ytotech.com/builds/sync"  # Vercel has no TeX
    memory_days: int = 1  # chat memory is deleted after this many days without use
    refresh_cooldown_hours: float = 6.0

    # Hybrid search tuning.
    search_candidates: int = 100  # candidates taken from each retriever before fusion
    rrf_k: int = 60  # standard reciprocal-rank-fusion constant


@lru_cache
def get_settings() -> Settings:
    return Settings()
