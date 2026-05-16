from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Phase 0 bootstrap: db/redis/auth only. AI/LLM/S3 added in Phase 2/4."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "aBridgeAI API"
    environment: str = "local"

    database_url: str = "postgresql+psycopg://abridgeai:abridgeai@localhost:5432/abridgeai"
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    db_pool_recycle_seconds: int = Field(default=1800, ge=0)

    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = Field(default="dev-only-change-me", min_length=16)
    access_token_ttl_seconds: int = 15 * 60
    session_ttl_seconds: int = 30 * 24 * 60 * 60

    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None

    # Observability (T0.24). Console default keeps `pytest` output human-readable
    # in dev; CI + production set LOG_FORMAT=json so log aggregators (Loki,
    # Datadog, CloudWatch) can parse JSON keys.
    log_format: Literal["json", "console"] = "console"
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
