"""DEV seed script — voice interview demo data + JWT printer.

Creates all rows needed to run a LiveKit voice interview end-to-end
locally, then prints teacher / student JWTs.

Idempotent: every INSERT uses ON CONFLICT DO NOTHING or DO UPDATE so the
script is safe to re-run. Stable UUIDs keep the output deterministic.

Usage::

    cd /path/to/co4029-backend
    uv run python scripts/seed_voice_interview_demo.py

Access rule discovered (learner.py:94-158):
    GET /interview-configs/{id} only requires:
    1. Valid JWT (auth_sessions row must exist for the session_id in JWT)
    2. Config status = 'published'
    No RBAC / enrollment check on this endpoint.
    Questions returned only have review_status = 'approved'.

    POST /interview-configs/{id}/sessions — same: only needs published config.
    No course enrollment check at the route layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# ---------------------------------------------------------------------------
# Stable demo IDs (deterministic across re-runs)
# ---------------------------------------------------------------------------
SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"

ORG_ID          = "00000000-0000-0000-0000-000000000020"
ORG_UNIT_ID     = "00000000-0000-0000-0000-000000000030"
COURSE_ID       = "00000000-0000-0000-0000-000000000040"
MODULE_ID       = "00000000-0000-0000-0000-000000000050"

TEACHER_ID      = "00000000-0000-0000-0000-000000000011"
STUDENT_ID      = "00000000-0000-0000-0000-000000000010"

# Interview config + question — stable but distinct from test UUIDs
CONFIG_ID       = "00000000-0000-0000-0001-000000000001"
QUESTION_ID     = "00000000-0000-0000-0001-000000000002"

# Auth session IDs (stable so JWTs are consistent between re-runs)
TEACHER_SESSION_ID = "00000000-0000-0000-0002-000000000001"
STUDENT_SESSION_ID = "00000000-0000-0000-0002-000000000002"

COURSE_SLUG = "voice-demo-course"


def _async_db_url(url: str) -> str:
    """Ensure URL uses psycopg async driver."""
    if "+psycopg_async" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return url


async def _seed(session: AsyncSession) -> None:
    """Run all idempotent upserts in a single transaction."""

    # ------------------------------------------------------------------
    # 1. Ensure permission/role catalog (mirrors conftest._ensure_catalog_seeded)
    # ------------------------------------------------------------------
    from abridgeai.access_control.permissions.loader import load_catalog, load_role_seeds

    catalog = load_catalog()
    role_seeds = load_role_seeds(catalog)

    # System user (audit FK anchor — same as migration 0004)
    await session.execute(
        text(
            "INSERT INTO users (id, primary_email, status, created_at, updated_at) "
            "VALUES (CAST(:id AS uuid), :email, 'inactive', NOW(), NOW()) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": SYSTEM_USER_ID, "email": "system@abridgeai.local"},
    )

    for perm in catalog.permissions:
        await session.execute(
            text(
                "INSERT INTO permissions (id, code, name, description, created_at, updated_at) "
                "VALUES (uuid_generate_v4(), :code, :name, :desc, NOW(), NOW()) "
                "ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING"
            ),
            {"code": perm.code, "name": perm.code, "desc": perm.description},
        )

    for role in role_seeds.roles:
        await session.execute(
            text(
                "INSERT INTO roles (id, code, name, is_system_role, created_at, updated_at) "
                "VALUES (uuid_generate_v4(), :code, :name, TRUE, NOW(), NOW()) "
                "ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING"
            ),
            {"code": role.code, "name": role.name},
        )
        for perm_code in role.permissions:
            await session.execute(
                text(
                    "INSERT INTO role_permissions (role_id, permission_id, created_at) "
                    "SELECT r.id, p.id, NOW() "
                    "FROM roles r JOIN permissions p ON p.code = :perm_code "
                    "WHERE r.code = :role_code "
                    "ON CONFLICT (role_id, permission_id) DO NOTHING"
                ),
                {"role_code": role.code, "perm_code": perm_code},
            )

    # ------------------------------------------------------------------
    # 2. Organization + org unit
    # ------------------------------------------------------------------
    await session.execute(
        text(
            "INSERT INTO organizations (id, slug, name, status) "
            "VALUES (:id, :slug, :name, 'active') "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ORG_ID, "slug": "demo-org", "name": "Demo Organization"},
    )
    await session.execute(
        text(
            "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
            "VALUES (:id, :org_id, 'department', 'Demo Dept', 'DEMO-DEPT') "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ORG_UNIT_ID, "org_id": ORG_ID},
    )

    # ------------------------------------------------------------------
    # 3. Teacher + student users with profiles
    # ------------------------------------------------------------------
    for uid, email, fname, lname, display in [
        (TEACHER_ID, "demo-teacher@abridgeai.local", "Demo", "Teacher", "Demo Teacher"),
        (STUDENT_ID, "demo-student@abridgeai.local", "Demo", "Student", "Demo Student"),
    ]:
        await session.execute(
            text(
                "INSERT INTO users (id, primary_email, status) "
                "VALUES (:id, :email, 'active') "
                "ON CONFLICT (id) DO UPDATE SET primary_email = EXCLUDED.primary_email, status = EXCLUDED.status"
            ),
            {"id": uid, "email": email},
        )
        await session.execute(
            text(
                "INSERT INTO user_profiles (user_id, given_name, family_name, display_name) "
                "VALUES (:uid, :fn, :ln, :dn) "
                "ON CONFLICT (user_id) DO UPDATE SET given_name=EXCLUDED.given_name, "
                "family_name=EXCLUDED.family_name, display_name=EXCLUDED.display_name"
            ),
            {"uid": uid, "fn": fname, "ln": lname, "dn": display},
        )

    # ------------------------------------------------------------------
    # 4. Course (must exist before role assignments reference course_id)
    # ------------------------------------------------------------------
    await session.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
            "VALUES (:id, :oid, :owner, :slug, :title, 'published') "
            "ON CONFLICT (id) DO UPDATE SET status = 'published'"
        ),
        {
            "id": COURSE_ID,
            "oid": ORG_ID,
            "owner": TEACHER_ID,
            "slug": COURSE_SLUG,
            "title": "Voice Demo Course",
        },
    )

    # ------------------------------------------------------------------
    # 5. Organization memberships
    # Unique index is partial (WHERE deleted_at IS NULL) on (user_id, org_id, org_unit_id)
    # so we use WHERE NOT EXISTS instead of ON CONFLICT.
    # ------------------------------------------------------------------
    for uid in (TEACHER_ID, STUDENT_ID):
        await session.execute(
            text(
                "INSERT INTO organization_memberships (id, user_id, organization_id, org_unit_id, status) "
                "SELECT gen_random_uuid(), :uid, :oid, :ouid, 'active' "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM organization_memberships "
                "  WHERE user_id = :uid AND organization_id = :oid AND org_unit_id = :ouid "
                "  AND deleted_at IS NULL"
                ")"
            ),
            {"uid": uid, "oid": ORG_ID, "ouid": ORG_UNIT_ID},
        )

    # ------------------------------------------------------------------
    # 6. Role assignments (mirrors conftest roles.yaml)
    # Unique index is also partial (WHERE deleted_at IS NULL) — use WHERE NOT EXISTS.
    # ------------------------------------------------------------------
    # Teacher: course-scoped teacher role
    await session.execute(
        text(
            "INSERT INTO user_role_assignments "
            "(user_id, role_id, scope_kind, organization_id, course_id) "
            "SELECT :uid, r.id, 'course', :oid, :cid "
            "FROM roles r WHERE r.code = 'teacher' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM user_role_assignments a "
            "  WHERE a.user_id = :uid AND a.role_id = r.id "
            "  AND a.scope_kind = 'course' AND a.course_id = :cid "
            "  AND a.deleted_at IS NULL"
            ")"
        ),
        {"uid": TEACHER_ID, "oid": ORG_ID, "cid": COURSE_ID},
    )
    # Student: org-scoped student role
    await session.execute(
        text(
            "INSERT INTO user_role_assignments "
            "(user_id, role_id, scope_kind, organization_id) "
            "SELECT :uid, r.id, 'organization', :oid "
            "FROM roles r WHERE r.code = 'student' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM user_role_assignments a "
            "  WHERE a.user_id = :uid AND a.role_id = r.id "
            "  AND a.scope_kind = 'organization' AND a.organization_id = :oid "
            "  AND a.deleted_at IS NULL"
            ")"
        ),
        {"uid": STUDENT_ID, "oid": ORG_ID},
    )

    # ------------------------------------------------------------------
    # 7. Module (after course)
    # ------------------------------------------------------------------
    await session.execute(
        text(
            "INSERT INTO modules (id, course_id, title, position, status) "
            "VALUES (:id, :cid, :title, 1, 'published') "
            "ON CONFLICT (id) DO UPDATE SET status = 'published'"
        ),
        {"id": MODULE_ID, "cid": COURSE_ID, "title": "Voice Demo Module"},
    )

    # ------------------------------------------------------------------
    # 8. Course enrollment (student)
    # ------------------------------------------------------------------
    await session.execute(
        text(
            "INSERT INTO course_enrollments (id, course_id, student_id, status, source) "
            "VALUES (gen_random_uuid(), :cid, :sid, 'active', 'manual') "
            "ON CONFLICT (course_id, student_id) DO UPDATE SET status = 'active'"
        ),
        {"cid": COURSE_ID, "sid": STUDENT_ID},
    )

    # ------------------------------------------------------------------
    # 9. InterviewConfig (supported_modes='voice', status='published')
    # ------------------------------------------------------------------
    await session.execute(
        text(
            "INSERT INTO interview_configs "
            "(id, course_id, module_id, title, status, supported_modes, time_limit_minutes, "
            "created_by, updated_by, created_at, updated_at) "
            "VALUES (:id, :cid, :mid, :title, 'published', 'voice', 10, "
            "CAST(:teacher AS uuid), CAST(:teacher AS uuid), NOW(), NOW()) "
            "ON CONFLICT (id) DO UPDATE SET "
            "status = 'published', supported_modes = 'voice', time_limit_minutes = 10"
        ),
        {
            "id": CONFIG_ID,
            "cid": COURSE_ID,
            "mid": MODULE_ID,
            "title": "Voice demo",
            "teacher": TEACHER_ID,
        },
    )

    # ------------------------------------------------------------------
    # 10. InterviewQuestion (review_status='approved' so it shows to student)
    # ------------------------------------------------------------------
    await session.execute(
        text(
            "INSERT INTO interview_questions "
            "(id, interview_config_id, position, question_type, prompt_text, "
            "review_status, ai_generated, created_by, updated_by, created_at, updated_at) "
            "VALUES (:id, :cfg, 1, 'behavioral', "
            ":prompt, 'approved', FALSE, "
            "CAST(:teacher AS uuid), CAST(:teacher AS uuid), NOW(), NOW()) "
            "ON CONFLICT (id) DO UPDATE SET "
            "review_status = 'approved', prompt_text = EXCLUDED.prompt_text"
        ),
        {
            "id": QUESTION_ID,
            "cfg": CONFIG_ID,
            "prompt": "Tell me about a project you built from scratch.",
            "teacher": TEACHER_ID,
        },
    )

    # ------------------------------------------------------------------
    # 11. Auth sessions (needed so JWTs validate against the DB)
    #     Use a stable refresh_token_hash per user (hash of session UUID).
    # ------------------------------------------------------------------
    for user_id, sess_id in [(TEACHER_ID, TEACHER_SESSION_ID), (STUDENT_ID, STUDENT_SESSION_ID)]:
        token_hash = hashlib.sha256(sess_id.encode()).hexdigest()
        await session.execute(
            text(
                "INSERT INTO auth_sessions "
                "(id, user_id, refresh_token_hash, expires_at) "
                "VALUES (CAST(:sid AS uuid), CAST(:uid AS uuid), :hash, NOW() + INTERVAL '30 days') "
                "ON CONFLICT (id) DO UPDATE SET "
                "expires_at = NOW() + INTERVAL '30 days', revoked_at = NULL"
            ),
            {"sid": sess_id, "uid": user_id, "hash": token_hash},
        )

    await session.commit()


def _mint_jwt(user_id: str, session_id: str) -> str:
    """Mint a long-lived dev JWT (30 days) for a stable session ID."""
    from abridgeai.core.security import create_access_token

    return create_access_token(
        user_id=UUID(user_id),
        session_id=UUID(session_id),
        expires_delta=timedelta(days=30),
    )


async def main() -> None:
    from abridgeai.core.config import get_settings

    settings = get_settings()

    # Use async driver (mirrors conftest pattern)
    db_url = settings.database_url
    if "+psycopg_async" not in db_url:
        db_url = _async_db_url(db_url)

    engine = create_async_engine(db_url, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async with Session() as session:
        await _seed(session)

    await engine.dispose()

    # Mint JWTs (stable session IDs so tokens are deterministic)
    teacher_jwt = _mint_jwt(TEACHER_ID, TEACHER_SESSION_ID)
    student_jwt  = _mint_jwt(STUDENT_ID,  STUDENT_SESSION_ID)

    print("\n" + "=" * 70)
    print("VOICE INTERVIEW DEMO SEED — SUCCESS")
    print("=" * 70)
    print(f"teacher_id:          {TEACHER_ID}")
    print(f"student_id:          {STUDENT_ID}")
    print(f"course_id:           {COURSE_ID}")
    print(f"course_slug:         {COURSE_SLUG}")
    print(f"module_id:           {MODULE_ID}")
    print(f"interview_config_id: {CONFIG_ID}")
    print(f"  status:            published")
    print(f"  supported_modes:   voice")
    print(f"  time_limit_min:    10")
    print()
    print(f"TEACHER_JWT:\n{teacher_jwt}")
    print()
    print(f"STUDENT_JWT:\n{student_jwt}")
    print()
    print("To verify (requires running API on localhost:8000):")
    print(f"  curl -s localhost:8000/api/v1/interview-configs/{CONFIG_ID} \\")
    print(f'    -H "Authorization: Bearer $STUDENT_JWT" | python3 -m json.tool')
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
