from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.core.security import create_access_token

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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


@pytest_asyncio.fixture(scope="session")
async def test_engine() -> AsyncEngine:
    settings = get_settings()
    engine = create_async_engine(_async_url(settings.database_url), pool_pre_ping=True)
    yield engine
    await engine.dispose()


def _load_yaml(name: str) -> dict:
    with (FIXTURES_DIR / name).open() as f:
        return yaml.safe_load(f)


async def _purge(session: AsyncSession, users_data: dict, roles_data: dict) -> None:
    user_ids = [u["id"] for u in users_data["users"]]
    role_codes = [r["code"] for r in roles_data["roles"]]
    org_id = roles_data["test_organization"]["id"]
    org_unit_id = roles_data["test_org_unit"]["id"]
    course_id = roles_data["test_course"]["id"]

    await session.execute(
        text("DELETE FROM user_role_assignments WHERE user_id = ANY(:ids)"),
        {"ids": user_ids},
    )
    await session.execute(
        text("DELETE FROM user_profiles WHERE user_id = ANY(:ids)"),
        {"ids": user_ids},
    )
    await session.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
    await session.execute(text("DELETE FROM org_units WHERE id = :id"), {"id": org_unit_id})
    await session.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": user_ids})
    await session.execute(text("DELETE FROM roles WHERE code = ANY(:codes)"), {"codes": role_codes})
    await session.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


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


async def _insert_roles(session: AsyncSession, roles: list[dict]) -> dict[str, str]:
    code_to_id: dict[str, str] = {}
    for role in roles:
        result = await session.execute(
            text(
                "INSERT INTO roles (code, name, is_system_role) "
                "VALUES (:code, :name, :is_system_role) RETURNING id"
            ),
            role,
        )
        code_to_id[role["code"]] = str(result.scalar_one())
    return code_to_id


async def _insert_assignments(
    session: AsyncSession, assignments: list[dict], role_id_by_code: dict[str, str]
) -> None:
    for a in assignments:
        await session.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id, org_unit_id, course_id) "
                "VALUES (:user_id, :role_id, :scope_kind, :organization_id, :org_unit_id, :course_id)"
            ),
            {
                "user_id": a["user_id"],
                "role_id": role_id_by_code[a["role_code"]],
                "scope_kind": a["scope_kind"],
                "organization_id": a.get("organization_id"),
                "org_unit_id": a.get("org_unit_id"),
                "course_id": a.get("course_id"),
            },
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

    async with test_engine.begin() as conn:
        session = AsyncSession(bind=conn, expire_on_commit=False)
        await _purge(session, users_data, roles_data)
        await _insert_organization(session, org)
        await _insert_org_unit(session, org_unit, org["id"])
        await _insert_users(session, users_data["users"])
        await _insert_course(session, course, org["id"], admin_id)
        role_id_by_code = await _insert_roles(session, roles_data["roles"])
        await _insert_assignments(session, roles_data["assignments"], role_id_by_code)
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
