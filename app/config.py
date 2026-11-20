"""Runtime settings, read from environment variables (prefix JOBPILOT_) or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="JOBPILOT_", extra="ignore")

    # Required: there is deliberately no default, so a missing database URL fails loudly at startup.
    database_url: str

    # Must produce vectors of app.models.EMBEDDING_DIM dimensions (384 for bge-small).
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    sources_file: str = "sources.yaml"
    # A posting not seen for this many days is marked inactive (for sources that are not full snapshots).
    expire_after_days: int = 14
    http_timeout_s: float = 30.0
    user_agent: str = "JobPilot/0.1 (+https://github.com/GaneshPokharkar/jobpilot)"

    # LLM (skill extraction and, later, the agent). Optional so ingestion and search run without it;
    # endpoints that need it return 503 until it is set.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"

    # Hybrid search tuning.
    search_candidates: int = 100  # candidates taken from each retriever before fusion
    rrf_k: int = 60  # standard reciprocal-rank-fusion constant


@lru_cache
def get_settings() -> Settings:
    return Settings()
