from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

# Side-effect import: completes the ORM registry so string-referenced
# cross-feature relationships (ModuleItem -> "Quiz"/"InterviewConfig")
# resolve even when a single test file is run in isolation.
import abridgeai.core.db.all_models  # noqa: F401
from tests.support.db_graph import hard_delete_graph as _hard_delete_graph
from abridgeai.access_control.permissions.loader import load_catalog, load_role_seeds
from abridgeai.core.config import get_settings
from abridgeai.core.security import create_access_token

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Stable system user (mirrors migration 0004). Re-inserted defensively by
# _ensure_catalog_seeded so a destructive migration round-trip mid-suite
# cannot leave audit FKs dangling.
_SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"
_SYSTEM_USER_EMAIL = "system@abridgeai.local"


@dataclass(frozen=True)
class SeededUsers:
    student_id: UUID
    teacher_id: UUID
    hod_id: UUID
    manager_id: UUID
    admin_id: UUID
    organization_id: UUID
    org_unit_id: UUID
    course_id: UUID


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _catalog_seed_after_reset() -> None:
    """Re-seed the migration-0004 platform catalog after a DB reset.

    The system user (``00000000-...-0001``), permissions and roles are
    seeded BY migration 0004, not by the test harness. A TRUNCATE-based
    test-DB reset (the documented recipe) wipes those rows while keeping
    ``alembic_version`` at head, so the migrations never re-run and every
    self-contained suite that inserts organizations (whose ``created_by``
    defaults to the system actor) or resolves roles by code fails on an
    otherwise clean database. Run the idempotent catalog seed once per
    session regardless of whether ``seeded_users`` is requested.
    """
    settings = get_settings()
    engine = create_async_engine(_async_url(settings.database_url), pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await _ensure_catalog_seeded(AsyncSession(bind=conn, expire_on_commit=False))
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def test_engine() -> AsyncEngine:
    # The Settings model swaps ``database_url`` to ``test_database_url``
    # automatically when pytest is imported (see
    # ``Settings._swap_to_test_db_under_pytest``). That single funnel covers
    # every per-file fixture that does ``create_async_engine(get_settings().database_url)``
    # — we don't have to rewrite ~30 such fixtures. If TEST_DATABASE_URL is
    # unset or identical to DATABASE_URL, Settings construction itself will
    # raise before we get here.
    settings = get_settings()
    engine = create_async_engine(_async_url(settings.database_url), pool_pre_ping=True)
    yield engine
    await engine.dispose()


def _load_yaml(name: str) -> dict:
    with (FIXTURES_DIR / name).open() as f:
        return yaml.safe_load(f)


async def _ensure_catalog_seeded(session: AsyncSession) -> None:
    catalog = load_catalog()
    role_seeds = load_role_seeds(catalog)

    await session.execute(
        text(
            "INSERT INTO users (id, primary_email, status, created_at, updated_at) "
            "VALUES (CAST(:user_id AS uuid), :email, 'inactive', NOW(), NOW()) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"user_id": _SYSTEM_USER_ID, "email": _SYSTEM_USER_EMAIL},
    )

    for perm in catalog.permissions:
        await session.execute(
            text(
                "INSERT INTO permissions (id, code, name, description, created_at, updated_at) "
                "VALUES (uuid_generate_v4(), :code, :name, :description, NOW(), NOW()) "
                "ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING"
            ),
            {"code": perm.code, "name": perm.code, "description": perm.description},
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

    for role in role_seeds.roles:
        permissions_codes = role.permissions
        if not isinstance(permissions_codes, list):
            msg = f"loader returned unexpanded sentinel for role {role.code!r}"
            raise RuntimeError(msg)
        for perm_code in permissions_codes:
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


async def _purge(session: AsyncSession, users_data: dict, roles_data: dict) -> None:
    user_ids = [u["id"] for u in users_data["users"]]
    org_id = roles_data["test_organization"]["id"]
    org_unit_id = roles_data["test_org_unit"]["id"]
    course_id = roles_data["test_course"]["id"]

    # 1. Remove all role assignments for test users (covers course-scoped ones too).
    await session.execute(
        text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
        {"ids": user_ids},
    )
    # 2. Remove memberships + profiles.
    await session.execute(
        text("DELETE FROM organization_memberships WHERE user_id = ANY(:ids)"),
        {"ids": user_ids},
    )
    await session.execute(
        text("DELETE FROM user_profiles WHERE user_id = ANY(:ids)"),
        {"ids": user_ids},
    )
    # 3. Delete the seeded test course + any extra courses tests may have created
    #    (e.g. via POST /teacher/courses). Both reference test users as owner_user_id.
    #    First remove any course-scoped role assignments that reference these courses
    #    (created by auto-assign-teacher logic) to avoid FK violations.
    #    Delete by course_id directly to handle all cases regardless of organization_id.
    await session.execute(
        text(
            "DELETE FROM user_role_assignments "
            "WHERE scope_kind = 'course' AND course_id IN ("
            "  SELECT id FROM courses WHERE organization_id = :org_id"
            ")"
        ),
        {"org_id": org_id},
    )
    # document_chunks reference courses (document_chunks_course_id_fkey) and are
    # created by material-processing tests. Purge any that reference the test
    # courses BEFORE deleting the courses themselves, else the course delete
    # trips a FK violation and every test sharing this session-scoped fixture
    # errors at setup.
    # Every FK targeting courses is NO ACTION (T0.14), so leftovers from any
    # crashed run sharing this database — enrollments, generation runs, whole
    # module/quiz subtrees — block the course delete and error EVERY test
    # using this session-scoped fixture at setup. The old fixed-list purge
    # (document_chunks only, then + enrollments, then + generation_runs...)
    # was whack-a-mole: each new FK into courses broke the suite again. This
    # walks the real FK graph instead, so a new dependent table is handled
    # the day its migration lands.
    doomed_courses = [
        str(row)
        for row in (
            await session.execute(
                text(
                    "SELECT id FROM courses WHERE id = :cid "
                    "OR owner_user_id = ANY(:ids) OR organization_id = :org_id"
                ),
                {"cid": course_id, "ids": user_ids, "org_id": org_id},
            )
        ).scalars()
    ]
    if doomed_courses:
        await _hard_delete_graph(session, "courses", doomed_courses)
    # 4. Users, then the org (which owns org_units, career_paths and whatever
    #    else a crashed run may have parked under it). Graph-deleted for the
    #    same reason as courses above: leftovers from other suites (audit
    #    rows, career paths) FK into these with NO ACTION and used to error
    #    every seeded-fixture test at setup.
    await _hard_delete_graph(session, "users", [str(u) for u in user_ids])
    await _hard_delete_graph(session, "organizations", [str(org_id)])
    del org_unit_id  # cleaned up as part of the organization subtree
    # Roles are now seeded permanently by migration 0004 and carry role_permissions
    # rows that block deletion. Tests bind to seeded role IDs via lookup_roles().


async def _insert_organization(session: AsyncSession, org: dict) -> None:
    await session.execute(
        text(
            "INSERT INTO organizations (id, slug, name, status) "
            "VALUES (:id, :slug, :name, 'active')"
        ),
        org,
    )


async def _insert_org_unit(session: AsyncSession, unit: dict, org_id: str) -> None:
    await session.execute(
        text(
            "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
            "VALUES (:id, :organization_id, :unit_type, :name, :code)"
        ),
        {**unit, "organization_id": org_id},
    )


async def _insert_users(session: AsyncSession, users: list[dict]) -> None:
    for user in users:
        await session.execute(
            text(
                "INSERT INTO users (id, primary_email, status) "
                "VALUES (:id, :primary_email, :status)"
            ),
            {"id": user["id"], "primary_email": user["primary_email"], "status": user["status"]},
        )
        profile = user["profile"]
        await session.execute(
            text(
                "INSERT INTO user_profiles (user_id, given_name, family_name, display_name) "
                "VALUES (:user_id, :given_name, :family_name, :display_name)"
            ),
            {"user_id": user["id"], **profile},
        )


async def _insert_course(session: AsyncSession, course: dict, org_id: str, owner_id: str) -> None:
    await session.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
            "VALUES (:id, :organization_id, :owner_user_id, :slug, :title, 'draft')"
        ),
        {
            "id": course["id"],
            "organization_id": org_id,
            "owner_user_id": owner_id,
            "slug": course["slug"],
            "title": course["title"],
        },
    )


async def _lookup_role_ids(session: AsyncSession, role_codes: list[str]) -> dict[str, str]:
    code_to_id: dict[str, str] = {}
    for code in role_codes:
        result = await session.execute(
            text("SELECT id FROM roles WHERE code = :code"),
            {"code": code},
        )
        code_to_id[code] = str(result.scalar_one())
    return code_to_id


async def _insert_assignments(
    session: AsyncSession, assignments: list[dict], role_id_by_code: dict[str, str]
) -> None:
    for a in assignments:
        await session.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id, org_unit_id, course_id, "
                "is_instructor, is_assistant) "
                "VALUES (:user_id, :role_id, :scope_kind, :organization_id, :org_unit_id, "
                ":course_id, :is_instructor, :is_assistant)"
            ),
            {
                "user_id": a["user_id"],
                "role_id": role_id_by_code[a["role_code"]],
                "scope_kind": a["scope_kind"],
                "organization_id": a.get("organization_id"),
                "org_unit_id": a.get("org_unit_id"),
                "course_id": a.get("course_id"),
                # The seed catalog's course rows are the ongoing Course
                # Instructor assignment the wider suite depends on.
                "is_instructor": a.get("is_instructor", a.get("scope_kind") == "course"),
                "is_assistant": a.get("is_assistant", False),
            },
        )


async def _insert_memberships(
    session: AsyncSession,
    *,
    org_id: str,
    user_ids: list[str],
    org_unit_id: str | None,
) -> None:
    """Seed organization_memberships rows for the test cohort.

    Non-admin users (student/teacher/hod/manager) belong to the test
    organization via membership -- this is the source of truth that
    ``access_control.api.public.get_user_primary_org`` consults to
    resolve a user's primary org. The admin user (scope=global) is
    intentionally excluded: platform admins have no implicit primary
    org and must use endpoints with explicit org_id.
    """
    for uid in user_ids:
        await session.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, org_unit_id, status) "
                "VALUES (gen_random_uuid(), :user_id, :org_id, :org_unit_id, 'active')"
            ),
            {"user_id": uid, "org_id": org_id, "org_unit_id": org_unit_id},
        )


@pytest_asyncio.fixture(scope="session")
async def seeded_users(test_engine: AsyncEngine) -> SeededUsers:
    users_data = _load_yaml("users.yaml")
    roles_data = _load_yaml("roles.yaml")

    org = roles_data["test_organization"]
    org_unit = roles_data["test_org_unit"]
    course = roles_data["test_course"]
    admin_id = next(
        u["id"] for u in users_data["users"] if u["primary_email"].endswith("admin@abridgeai.local")
    )
    member_user_ids = [u["id"] for u in users_data["users"] if u["id"] != admin_id]

    async with test_engine.begin() as conn:
        session = AsyncSession(bind=conn, expire_on_commit=False)
        await _ensure_catalog_seeded(session)
        await _purge(session, users_data, roles_data)
        await _insert_organization(session, org)
        await _insert_org_unit(session, org_unit, org["id"])
        await _insert_users(session, users_data["users"])
        await _insert_course(session, course, org["id"], admin_id)
        role_id_by_code = await _lookup_role_ids(session, [r["code"] for r in roles_data["roles"]])
        await _insert_assignments(session, roles_data["assignments"], role_id_by_code)
        await _insert_memberships(
            session,
            org_id=org["id"],
            user_ids=member_user_ids,
            org_unit_id=org_unit["id"],
        )
        await session.flush()

    by_email = {u["primary_email"]: UUID(u["id"]) for u in users_data["users"]}
    return SeededUsers(
        student_id=by_email["test-student@abridgeai.local"],
        teacher_id=by_email["test-teacher@abridgeai.local"],
        hod_id=by_email["test-hod@abridgeai.local"],
        manager_id=by_email["test-manager@abridgeai.local"],
        admin_id=by_email["test-admin@abridgeai.local"],
        organization_id=UUID(org["id"]),
        org_unit_id=UUID(org_unit["id"]),
        course_id=UUID(course["id"]),
    )


def _token(user_id: UUID) -> str:
    return create_access_token(user_id=user_id, session_id=uuid4())


@pytest.fixture
def student_token(seeded_users: SeededUsers) -> str:
    return _token(seeded_users.student_id)


@pytest.fixture
def teacher_token(seeded_users: SeededUsers) -> str:
    return _token(seeded_users.teacher_id)


@pytest.fixture
def hod_token(seeded_users: SeededUsers) -> str:
    return _token(seeded_users.hod_id)


@pytest.fixture
def manager_token(seeded_users: SeededUsers) -> str:
    return _token(seeded_users.manager_id)


@pytest.fixture
def admin_token(seeded_users: SeededUsers) -> str:
    return _token(seeded_users.admin_id)


@pytest_asyncio.fixture(autouse=True)
async def _reset_global_async_clients() -> AsyncIterator[None]:
    """Reset process-wide async singletons around every test.

    ``core.cache.client.get_cache`` lazily creates ONE Redis client for the
    process. pytest-asyncio gives each test its own event loop, so a client
    created on test A's loop is reused on test B's loop and redis-py raises
    ``got Future attached to a different loop`` — which is why any file whose
    pipeline touched ``publish_progress`` failed from the second async test
    onward. Recreating the client per test is cheap (creation is lazy and
    most tests never touch Redis at all).
    """
    from abridgeai.core.cache.client import close_cache, reset_cache_client_for_tests

    reset_cache_client_for_tests()
    yield
    try:
        await close_cache()
    except Exception:  # noqa: BLE001 -- teardown must never mask the test result
        reset_cache_client_for_tests()
