from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Class-level marker prefix for the development-only JWT default. Any secret
# starting with this prefix is rejected when ``environment == "production"``.
# We check the prefix (rather than the full literal "dev-only-change-me") so
# operators can't sneak the dev default past the validator with trivial
# variations like "dev-only-change-me-2" while still permitting custom dev
# secrets that explicitly opt into the "dev-" namespace in non-prod envs.
_DEV_JWT_SECRET_PREFIX = "dev-"  # noqa: S105  # nosec B105


# FR-12 whitelist for ``LLM_EXTRA_HEADERS_JSON``. Disallowed keys
# (case-insensitive) are dropped with a single WARN log so operators notice
# typos without a hard startup failure. Authentication-bearing headers are
# forbidden outright (they would leak credentials or override the configured
# Bearer token).
ALLOWED_EXTRA_HEADERS: frozenset[str] = frozenset(
    h.lower() for h in {"HTTP-Referer", "X-Title", "User-Agent"}
)
FORBIDDEN_EXTRA_HEADERS: frozenset[str] = frozenset(
    h.lower() for h in {"Authorization", "Cookie", "Set-Cookie", "Proxy-Authorization"}
)


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

    # T0.28 -- CORS allowed origins (comma-separated, e.g.
    # "http://localhost:5173,https://app.abridgeai.com"). Empty string means
    # CORS middleware is not mounted; cross-origin requests get the normal
    # response (browsers will block them by default).
    allowed_origins: str = ""

    database_url: str = "postgresql+psycopg://abridgeai:abridgeai@localhost:5432/abridgeai"
    # Separate database for the pytest suite. MUST point at a throwaway
    # database (created via `docker compose exec postgres createdb abridgeai_test`
    # or similar). When pytest is imported, the ``database_url`` field is
    # automatically rewritten to point at this URL via a model validator —
    # see ``_swap_to_test_db_under_pytest`` below. This means every test
    # fixture that reaches into ``get_settings().database_url`` lands on
    # the test DB without per-file fixture updates.
    test_database_url: str = ""
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    db_pool_recycle_seconds: int = Field(default=1800, ge=0)

    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = Field(default="dev-only-change-me", min_length=32)
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

    knowledge_graph_enabled: bool = False
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = Field(default=50, ge=1, le=1000)

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_default_tier: Literal["small", "standard", "large"] = "standard"
    llm_model_small: str = "gpt-4o-mini"
    llm_model_standard: str = "gpt-4o-mini"
    llm_model_large: str = "gpt-4o"
    llm_timeout_seconds: float = 60.0
    llm_extra_headers_json: str | None = None

    # Per-role overrides; None means "fall back to the tier mapping".
    llm_model_extraction: str | None = None
    llm_model_enrichment: str | None = None
    llm_model_ideation: str | None = None
    llm_model_generation: str | None = None
    llm_model_validation: str | None = None
    llm_model_chunking_enrichment: str | None = None
    llm_model_kg_extraction: str | None = None
    llm_model_stt: str | None = None
    llm_model_vision: str | None = None

    # Derived field, populated by ``_populate_extra_headers`` from
    # ``llm_extra_headers_json``. Always a dict (possibly empty).
    llm_extra_headers: dict[str, str] = Field(default_factory=dict)

    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_timeout_seconds: float = 30.0

    # Voyage rerank-2.5 (Phase 4 contextual-RAG upgrade). 200M tokens free
    # tier covers our scale; when the key is unset rerank is skipped and
    # retrieval falls back to the MMR-only path.
    voyage_api_key: SecretStr | None = None
    voyage_base_url: str = "https://api.voyageai.com/v1"
    voyage_rerank_model: str = "rerank-2.5"
    voyage_rerank_timeout_seconds: float = 15.0

    # Phase 3 — Contextual BM25 hybrid search (Anthropic guide). When
    # ``hybrid_bm25_enabled`` is True the retriever runs vector_search
    # AND bm25_search at recall_k each, then fuses with weighted
    # Reciprocal Rank Fusion (semantic_weight + bm25_weight should sum
    # to 1.0). Defaults mirror the Anthropic cookbook (0.8 / 0.2 with
    # recall=150). The knob defaults OFF so existing deployments behave
    # unchanged until operators flip it; flipping is a runtime env var,
    # no code change.
    hybrid_bm25_enabled: bool = False
    hybrid_recall_k: int = Field(default=150, ge=10, le=1000)
    hybrid_semantic_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    hybrid_bm25_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    # T2.3: S3 / Garage object storage. ``aws_endpoint_url`` is the
    # server-side hostname (None → AWS default); ``aws_public_endpoint_url``
    # is the browser-reachable hostname for presigned URLs (falls back to
    # the internal endpoint). Path-style addressing is forced when an
    # endpoint is set (Garage requires it); virtual-host style is used
    # otherwise (AWS S3 default).
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    aws_endpoint_url: str | None = None
    aws_public_endpoint_url: str | None = None
    aws_region: str = "us-east-1"
    s3_url_ttl_seconds: int = Field(default=3600, ge=60, le=24 * 60 * 60)
    s3_bucket_name: str = "abridgeai-materials"

    audio_extraction_local: bool = False
    image_ocr_provider: Literal["tesseract", "llm_vision"] = "tesseract"
    ffmpeg_path: str = "ffmpeg"
    whisper_model: str = "whisper-1"
    video_frame_sample_fps: float = Field(default=1.0, gt=0, le=30)

    # LiveKit voice-interview (Phase 1+). Target is LiveKit Cloud for dev +
    # initial prod; ``livekit_ws_url`` is the project WS endpoint
    # (wss://<project>.livekit.cloud). Voice mode is gated OFF by default so
    # existing deployments are unaffected until operators flip the flag AND
    # provide credentials. The key/secret are SecretStr (never logged).
    interview_voice_enabled: bool = False
    livekit_ws_url: str | None = None
    livekit_api_key: SecretStr | None = None
    livekit_api_secret: SecretStr | None = None
    livekit_agent_name: str = "interview-agent"
    interview_voice_token_ttl_seconds: int = Field(default=900, ge=60, le=24 * 60 * 60)
    # Safety net for voice sessions whose config has no ``time_limit_minutes``:
    # the stale-session sweep finalises them once they have been idle (no new
    # message) for this many minutes, so a dropped/abandoned session can never
    # stay ``in_progress`` forever. Time-limited sessions are unaffected.
    interview_voice_idle_timeout_minutes: int = Field(default=30, ge=1, le=24 * 60)
    # FR-4.5 / FR-5.3 — server-side lesson-gating enforcement (prerequisites,
    # SR coverage threshold τ, interview-pass locks). Emergency off-switch:
    # set LESSON_GATING_ENFORCED=false to stop blocking learner reads while
    # keeping the lock indicators visible in dashboards.
    lesson_gating_enforced: bool = True

    @model_validator(mode="after")
    def _require_livekit_creds_when_voice_enabled(self) -> Settings:
        """When voice interviews are enabled, all three LiveKit credentials
        must be present. Fail fast at startup rather than at first token mint."""
        if not self.interview_voice_enabled:
            return self
        missing = [
            name
            for name, value in (
                ("LIVEKIT_WS_URL", self.livekit_ws_url),
                ("LIVEKIT_API_KEY", self.livekit_api_key),
                ("LIVEKIT_API_SECRET", self.livekit_api_secret),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "INTERVIEW_VOICE_ENABLED=true requires "
                f"{', '.join(missing)} to be set (LiveKit credentials)."
            )
        return self

    @model_validator(mode="after")
    def _reject_dev_jwt_secret_in_production(self) -> Settings:
        if self.environment == "production" and self.jwt_secret_key.startswith(
            _DEV_JWT_SECRET_PREFIX
        ):
            raise ValueError(
                "jwt_secret_key starting with 'dev-' is forbidden when "
                "environment='production'; provide a strong secret "
                "(>=32 bytes of entropy) via JWT_SECRET_KEY"
            )
        return self

    @model_validator(mode="after")
    def _swap_to_test_db_under_pytest(self) -> Settings:
        """When pytest is running, transparently rewrite ``database_url`` to
        ``test_database_url`` so every fixture (including per-file ones that
        do ``create_async_engine(get_settings().database_url)``) lands on
        the throwaway test DB.

        Past sessions leaked 276 ``@test.local`` users into the live DB
        before this guard existed. The check is single-funnel: pytest is
        never imported in the runtime, so this only fires under tests.
        """
        import sys

        if "pytest" not in sys.modules:
            return self

        test_url = (self.test_database_url or "").strip()
        if not test_url:
            raise ValueError(
                "TEST_DATABASE_URL is not set. Refusing to run pytest "
                "against the production database (would leak fixture "
                "users into live data). Create a throwaway DB and set "
                "TEST_DATABASE_URL in .env or your shell. Example:\n"
                "  docker compose exec postgres createdb -U abridgeai abridgeai_test\n"
                "  export TEST_DATABASE_URL='postgresql+psycopg://abridgeai:...@localhost:5433/abridgeai_test'"
            )
        if test_url == self.database_url:
            raise ValueError(
                "TEST_DATABASE_URL is identical to DATABASE_URL. Use a "
                "separate database — the test suite seeds disposable "
                "@test.local users that must not leak into production."
            )
        # Rewrite in-place so any code path that hits ``settings.database_url``
        # (whether via the global ``test_engine`` fixture or an ad-hoc one) is
        # automatically funnelled to the test DB.
        object.__setattr__(self, "database_url", test_url)
        return self

    @model_validator(mode="after")
    def _populate_extra_headers(self) -> Settings:
        """Parse, whitelist, and store the LLM extra-headers map (FR-12).

        Failure modes:
          * ``Authorization``/``Cookie``/``Set-Cookie``/``Proxy-Authorization``
            (case-insensitive) raise ``ConfigError`` — these would either leak
            credentials or override the configured Bearer token.
          * Invalid JSON raises ``ConfigError``.
          * Other unrecognized keys are dropped with a single WARN log so the
            operator notices typos without breaking startup.
        """
        # Lazy import to avoid a circular dependency: abridgeai.ai.llm imports
        # back into abridgeai.core.config for Settings + get_settings.
        from abridgeai.ai.llm.errors import ConfigError

        raw = self.llm_extra_headers_json
        if raw is None or raw == "":
            object.__setattr__(self, "llm_extra_headers", {})
            return self

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"LLM_EXTRA_HEADERS_JSON is not valid JSON: {exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise ConfigError("LLM_EXTRA_HEADERS_JSON must decode to a JSON object")

        cleaned: dict[str, str] = {}
        dropped: list[str] = []
        for key, value in parsed.items():
            lower_key = str(key).lower()
            if lower_key in FORBIDDEN_EXTRA_HEADERS:
                raise ConfigError(
                    f"LLM_EXTRA_HEADERS_JSON contains forbidden header {key!r}; "
                    f"authentication-bearing headers "
                    f"({sorted(FORBIDDEN_EXTRA_HEADERS)}) are not allowed"
                )
            if lower_key not in ALLOWED_EXTRA_HEADERS:
                dropped.append(str(key))
                continue
            cleaned[str(key)] = str(value)

        if dropped:
            logger.warning(
                "LLM_EXTRA_HEADERS_JSON: dropping disallowed header(s) %s; allowed keys are %s",
                dropped,
                sorted(ALLOWED_EXTRA_HEADERS),
            )

        object.__setattr__(self, "llm_extra_headers", cleaned)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
