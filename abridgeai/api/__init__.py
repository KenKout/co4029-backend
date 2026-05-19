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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from abridgeai.api.healthz import router as healthz_router
from abridgeai.core.config import get_settings
from abridgeai.core.observability.audit_log import AuditLogMiddleware
from abridgeai.core.security.headers import SecurityHeadersMiddleware
from abridgeai.features.access_control.routers import admin_router as access_control_admin_router
from abridgeai.features.admin.routers import (
    ai_costs_router as admin_ai_costs_router,
)
from abridgeai.features.admin.routers import (
    audit_router as admin_audit_router,
)
from abridgeai.features.admin.routers import (
    processing_router as admin_processing_router,
)
from abridgeai.features.admin.routers import (
    stats_router as admin_stats_router,
)
from abridgeai.features.admin.routers import (
    users_router as admin_users_router,
)
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
from abridgeai.features.enrollments.routers import (
    assignment_dept_router as enrollments_dept_router,
)
from abridgeai.features.enrollments.routers import (
    assignment_management_router as enrollments_management_router,
)
from abridgeai.features.enrollments.routers import (
    me_enrollments_router,
)
from abridgeai.features.identity.routers import (
    auth_router as identity_auth_router,
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
from abridgeai.features.materials.routers import (
    authoring_router as materials_authoring_router,
)
from abridgeai.features.materials.routers import (
    learner_router as materials_learner_router,
)
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
from abridgeai.features.spaced_repetition.routers import (
    learner_router as spaced_repetition_learner_router,
)
from abridgeai.features.spaced_repetition.routers import (
    teacher_router as spaced_repetition_teacher_router,
)

API_V1_PREFIX = "/api/v1"


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
    )

    settings = get_settings()

    # Phase 1 -- identity
    app.include_router(identity_auth_router, prefix=API_V1_PREFIX)
    app.include_router(identity_me_router, prefix=API_V1_PREFIX)
    app.include_router(identity_mfa_router, prefix=API_V1_PREFIX)
    app.include_router(identity_users_router, prefix=API_V1_PREFIX)

    # Phase 2 -- access control
    app.include_router(access_control_admin_router, prefix=API_V1_PREFIX)

    # Phase 3 -- courses
    app.include_router(courses_learner_router, prefix=API_V1_PREFIX)
    app.include_router(me_courses_router, prefix=API_V1_PREFIX)
    app.include_router(courses_authoring_router, prefix=API_V1_PREFIX)
    app.include_router(courses_assignment_router, prefix=API_V1_PREFIX)
    app.include_router(courses_administration_router, prefix=API_V1_PREFIX)

    # Phase 4 -- materials
    app.include_router(materials_learner_router, prefix=API_V1_PREFIX)
    app.include_router(materials_authoring_router, prefix=API_V1_PREFIX)

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
