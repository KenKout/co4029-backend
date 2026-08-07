"""FastAPI application factory (T7.6 -- Phase 7 final integration).

Composes the cross-feature router surface that ships with Phase 7. All
routers are mounted under ``/api/v1``; the routers themselves carry their
own per-feature prefix (e.g. ``/teacher``, ``/me/notifications``,
``/admin/stats``) so the wiring layer is purely a list of
``include_router`` calls.

Wiring philosophy (per Reconciliation §A1 / Phase 7 contract):

* Identity, access_control, courses, materials, quizzes, interviews are
  all already-shipped (Phases 1-6). Phase 7 adds enrollments, progress,
  career_paths, notifications, and admin.
* Cross-feature *event wiring* (e.g. enrollment-added -> dispatch
  notification) is left to the per-feature service layer; this factory
  only owns transport (HTTP) composition.
* Lifespan / DB pool / Redis / observability bootstrapping is intentionally
  deferred -- Phase 9 layers production middleware on top of this factory.
  Tests instantiate ``create_app()`` directly and override ``get_db`` /
  ``get_arq_pool`` with in-process fakes.

The factory is import-safe: importing this module does *not* open DB
connections, start ARQ pools, or read environment beyond what FastAPI's
constructor itself touches.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from abridgeai.api.healthz import router as healthz_router
from abridgeai.core.config import get_settings
from abridgeai.core.observability import get_logger
from abridgeai.core.observability.audit_log import AuditLogMiddleware
from abridgeai.core.security.headers import SecurityHeadersMiddleware
from abridgeai.features.access_control.routers import admin_router as access_control_admin_router
from abridgeai.features.access_control.routers import (
    organizations_router as access_control_organizations_router,
)
from abridgeai.features.admin.routers import (
    ai_costs_router as admin_ai_costs_router,
)
from abridgeai.features.admin.routers import (
    ai_pricing_router as admin_ai_pricing_router,
)
from abridgeai.features.admin.routers import (
    audit_router as admin_audit_router,
)
from abridgeai.features.admin.routers import (
    processing_router as admin_processing_router,
)
from abridgeai.features.admin.routers import (
    settings_router as admin_settings_router,
)
from abridgeai.features.admin.routers import (
    stats_router as admin_stats_router,
)
from abridgeai.features.admin.routers import (
    users_router as admin_users_router,
)
from abridgeai.features.admin.routers.processing import get_arq_pool as admin_get_arq_pool
from abridgeai.features.career_paths.routers import (
    authoring_management_router as career_paths_management_router,
)
from abridgeai.features.career_paths.routers import (
    authoring_teacher_router as career_paths_teacher_router,
)
from abridgeai.features.career_paths.routers import (
    career_paths_learner_router,
    me_career_enrollments_router,
)
from abridgeai.features.courses.routers import (
    administration_router as courses_administration_router,
)
from abridgeai.features.courses.routers import (
    assignment_router as courses_assignment_router,
)
from abridgeai.features.courses.routers import (
    authoring_router as courses_authoring_router,
)
from abridgeai.features.courses.routers import (
    learner_router as courses_learner_router,
)
from abridgeai.features.courses.routers import (
    me_courses_router,
)
from abridgeai.features.courses.routers.assignment import (
    get_arq_pool as courses_assignment_get_arq_pool,
)
from abridgeai.features.courses.routers.authoring import (
    get_arq_pool as courses_authoring_get_arq_pool,
)
from abridgeai.features.discussions.router import router as discussions_router
from abridgeai.features.enrollments.routers import (
    assignment_dept_router as enrollments_dept_router,
)
from abridgeai.features.enrollments.routers import (
    assignment_management_router as enrollments_management_router,
)
from abridgeai.features.enrollments.routers import (
    assignment_teacher_router as enrollments_teacher_router,
)
from abridgeai.features.enrollments.routers import (
    me_enrollments_router,
)
from abridgeai.features.enrollments.routers.assignment import (
    get_arq_pool as enrollments_assignment_get_arq_pool,
)
from abridgeai.features.identity.routers import (
    auth_router as identity_auth_router,
)
from abridgeai.features.identity.routers import (
    me_root_router as identity_me_root_router,
)
from abridgeai.features.identity.routers import (
    me_router as identity_me_router,
)
from abridgeai.features.identity.routers import (
    mfa_router as identity_mfa_router,
)
from abridgeai.features.identity.routers import (
    users_router as identity_users_router,
)
from abridgeai.features.interviews.routers import (
    authoring_router as interviews_authoring_router,
)
from abridgeai.features.interviews.routers import (
    learner_router as interviews_learner_router,
)
from abridgeai.features.interviews.routers.authoring import (
    get_arq_pool as interviews_authoring_get_arq_pool,
)
from abridgeai.features.interviews.routers.learner import (
    get_arq_pool as interviews_learner_get_arq_pool,
)
from abridgeai.features.materials.routers import (
    authoring_router as materials_authoring_router,
)
from abridgeai.features.materials.routers import (
    learner_router as materials_learner_router,
)
from abridgeai.features.materials.routers import (
    materials_upload_router,
)
from abridgeai.features.materials.routers.authoring import get_arq_pool as materials_get_arq_pool
from abridgeai.features.notifications.routers import (
    learner_router as notifications_learner_router,
)
from abridgeai.features.progress.routers import (
    authoring_router as progress_authoring_router,
)
from abridgeai.features.progress.routers import (
    learner_router as progress_learner_router,
)
from abridgeai.features.quizzes.routers import (
    authoring_router as quizzes_authoring_router,
)
from abridgeai.features.quizzes.routers import (
    learner_router as quizzes_learner_router,
)
from abridgeai.features.quizzes.routers.authoring import get_arq_pool as quizzes_get_arq_pool
from abridgeai.features.spaced_repetition.routers import (
    learner_router as spaced_repetition_learner_router,
)
from abridgeai.features.spaced_repetition.routers import (
    teacher_router as spaced_repetition_teacher_router,
)
from abridgeai.infrastructure.neo4j_schema import ensure_graph_schema

API_V1_PREFIX = "/api/v1"


async def _probe_embedding_health() -> None:
    """Non-fatal startup probe of the embedding provider.

    Quiz generation and material ingestion both depend on the embedding
    endpoint, but a misconfigured provider (invalid token, or a token that
    lacks access to the configured embedding model) currently only surfaces
    when a teacher clicks Generate and the background job fails minutes later
    with an opaque upstream error. This probe fires one tiny embedding call on
    boot and logs a LOUD operator warning if it fails, so a misprovisioned
    proxy is visible immediately in the startup log.

    It deliberately NEVER raises: an embedding outage must not take down login,
    courses, or the rest of the API. Best-effort visibility only.
    """
    from abridgeai.ai.llm.client import OpenAICompatibleClient
    from abridgeai.ai.llm.roles import LLMRole, binding_for

    log = get_logger(__name__)
    settings = get_settings()
    try:
        binding = binding_for(LLMRole.EMBEDDING, settings)
        client = OpenAICompatibleClient(binding)
        await client.embeddings(["healthcheck"], dimensions=settings.embedding_dimensions)
        log.info("embedding_health_ok", model=binding.model, base_url=binding.base_url)
    except Exception as exc:  # noqa: BLE001 -- must never crash startup
        log.error(
            "embedding_health_failed",
            error=str(exc),
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            hint=(
                "Quiz generation and material ingestion will fail. Check the "
                "embedding provider token has access to the configured model "
                "on the LLM proxy."
            ),
        )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))

    async def _provide_arq_pool() -> object:
        return pool

    app.dependency_overrides[admin_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[interviews_authoring_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[interviews_learner_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[materials_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[quizzes_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[courses_assignment_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[courses_authoring_get_arq_pool] = _provide_arq_pool
    app.dependency_overrides[enrollments_assignment_get_arq_pool] = _provide_arq_pool

    # Best-effort embedding provider probe (non-fatal — logs, never raises).
    await _probe_embedding_health()

    # Idempotent Neo4j constraints + indexes. The graph ran without either
    # until now, so every MERGE was a label scan and duplicate Concept nodes
    # were possible under concurrent ingest. Non-fatal like the probe above.
    await ensure_graph_schema()

    yield

    await pool.aclose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    All feature routers are mounted under :data:`API_V1_PREFIX`. The
    factory is deliberately side-effect-free at import time: tests can
    call it freely and override DB / ARQ dependencies before sending
    any traffic.
    """
    app = FastAPI(
        title="aBridgeAI API",
        description="Cross-feature API composing identity, courses, materials, "
        "quizzes, interviews, enrollments, progress, career paths, "
        "notifications, and admin surfaces.",
        version="0.7.0",
        lifespan=_lifespan,
    )

    settings = get_settings()

    # Phase 1 -- identity
    app.include_router(identity_auth_router, prefix=API_V1_PREFIX)
    app.include_router(identity_me_router, prefix=API_V1_PREFIX)
    app.include_router(identity_me_root_router, prefix=API_V1_PREFIX)
    app.include_router(identity_mfa_router, prefix=API_V1_PREFIX)
    app.include_router(identity_users_router, prefix=API_V1_PREFIX)

    # Phase 2 -- access control
    app.include_router(access_control_admin_router, prefix=API_V1_PREFIX)
    app.include_router(access_control_organizations_router, prefix=API_V1_PREFIX)

    # Phase 3 -- courses
    app.include_router(courses_learner_router, prefix=API_V1_PREFIX)
    app.include_router(me_courses_router, prefix=API_V1_PREFIX)
    app.include_router(courses_authoring_router, prefix=API_V1_PREFIX)
    app.include_router(courses_assignment_router, prefix=API_V1_PREFIX)
    app.include_router(courses_administration_router, prefix=API_V1_PREFIX)

    # Phase 4 -- materials
    app.include_router(materials_learner_router, prefix=API_V1_PREFIX)
    app.include_router(materials_authoring_router, prefix=API_V1_PREFIX)
    app.include_router(materials_upload_router, prefix=API_V1_PREFIX)

    # Lesson discussions (teacher topics + student comments)
    app.include_router(discussions_router, prefix=API_V1_PREFIX)

    # Phase 5 -- quizzes
    app.include_router(quizzes_learner_router, prefix=API_V1_PREFIX)
    app.include_router(quizzes_authoring_router, prefix=API_V1_PREFIX)

    # Phase 6 -- interviews
    app.include_router(interviews_learner_router, prefix=API_V1_PREFIX)
    app.include_router(interviews_authoring_router, prefix=API_V1_PREFIX)

    # Phase 7 -- enrollments (T7.1)
    app.include_router(me_enrollments_router, prefix=API_V1_PREFIX)
    app.include_router(enrollments_dept_router, prefix=API_V1_PREFIX)
    app.include_router(enrollments_management_router, prefix=API_V1_PREFIX)
    app.include_router(enrollments_teacher_router, prefix=API_V1_PREFIX)

    # Phase 7 -- progress (T7.2)
    app.include_router(progress_learner_router, prefix=API_V1_PREFIX)
    app.include_router(progress_authoring_router, prefix=API_V1_PREFIX)

    # Phase 7 -- career paths (T7.3)
    app.include_router(career_paths_learner_router, prefix=API_V1_PREFIX)
    app.include_router(me_career_enrollments_router, prefix=API_V1_PREFIX)
    app.include_router(career_paths_management_router, prefix=API_V1_PREFIX)
    app.include_router(career_paths_teacher_router, prefix=API_V1_PREFIX)

    # Phase 7 -- notifications (T7.4)
    app.include_router(notifications_learner_router, prefix=API_V1_PREFIX)

    # Phase 7 -- admin (T7.5)
    app.include_router(admin_stats_router, prefix=API_V1_PREFIX)
    app.include_router(admin_audit_router, prefix=API_V1_PREFIX)
    app.include_router(admin_processing_router, prefix=API_V1_PREFIX)
    app.include_router(admin_users_router, prefix=API_V1_PREFIX)

    # Phase 7.5 -- spaced repetition dashboards (T7.5.12)
    app.include_router(spaced_repetition_learner_router, prefix=API_V1_PREFIX)
    app.include_router(spaced_repetition_teacher_router, prefix=API_V1_PREFIX)

    # T0.27 -- AI cost dashboard (admin observability)
    app.include_router(admin_ai_costs_router, prefix=API_V1_PREFIX)

    # Admin-configurable AI model pricing (replaces hardcoded PRICE_TABLE)
    app.include_router(admin_ai_pricing_router, prefix=API_V1_PREFIX)
    app.include_router(admin_settings_router, prefix=API_V1_PREFIX)

    # T0.28 -- security headers (HSTS, CSP, X-Frame-Options, ...). Added
    # BEFORE the audit-log middleware in source order so it sits INSIDE the
    # audit-log layer at runtime (Starlette wraps later additions outermost),
    # which means the audit log records the final headers-decorated response.
    app.add_middleware(SecurityHeadersMiddleware, environment=settings.environment)

    # T0.28 -- CORS. Only mounted when ALLOWED_ORIGINS is set; otherwise no
    # CORS headers are emitted and browsers block cross-origin traffic by
    # default. Wildcard origins are intentionally NOT supported because we
    # set ``allow_credentials=True`` (browsers refuse "*" + credentials).
    allowed_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
        )

    # T0.23 -- HTTP audit log middleware (logs every request to http_audit_log)
    app.add_middleware(AuditLogMiddleware)

    # T0.26 -- health checks (mounted at root so K8s probes + load balancers
    # can hit /healthz, /healthz/deep, /readyz without the /api/v1 prefix).
    # Also mounted under /api/v1 so the SPA can probe via the same reverse-
    # proxy /api/v1 path (front-door /healthz at the public host serves the
    # SPA HTML, not the backend).
    app.include_router(healthz_router)
    app.include_router(healthz_router, prefix=API_V1_PREFIX)

    return app


__all__ = ["API_V1_PREFIX", "create_app"]
