from __future__ import annotations

import pytest
from conftest import SeededUsers
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.core.security import decode_access_token


@pytest.mark.asyncio
async def test_seed_users(test_engine: AsyncEngine, seeded_users: SeededUsers) -> None:
    async with AsyncSession(test_engine) as session:
        user_count = await session.scalar(
            text("SELECT COUNT(*) FROM users WHERE primary_email LIKE 'test-%@abridgeai.local'")
        )
        assert user_count == 5

        profile_count = await session.scalar(
            text(
                "SELECT COUNT(*) FROM user_profiles up "
                "JOIN users u ON u.id = up.user_id "
                "WHERE u.primary_email LIKE 'test-%@abridgeai.local'"
            )
        )
        assert profile_count == 5

        assignment_count = await session.scalar(
            text(
                "SELECT COUNT(*) FROM user_role_assignments ra "
                "JOIN users u ON u.id = ra.user_id "
                "WHERE u.primary_email LIKE 'test-%@abridgeai.local'"
            )
        )
        assert assignment_count == 5


@pytest.mark.parametrize(
    ("token_fixture", "attr"),
    [
        ("student_token", "student_id"),
        ("teacher_token", "teacher_id"),
        ("hod_token", "hod_id"),
        ("manager_token", "manager_id"),
        ("admin_token", "admin_id"),
    ],
)
def test_each_token_decodes(
    token_fixture: str, attr: str, seeded_users: SeededUsers, request: pytest.FixtureRequest
) -> None:
    token = request.getfixturevalue(token_fixture)
    payload = decode_access_token(token)
    assert payload.sub == getattr(seeded_users, attr)


@pytest.mark.asyncio
async def test_hod_org_unit_scope(test_engine: AsyncEngine, seeded_users: SeededUsers) -> None:
    async with AsyncSession(test_engine) as session:
        row = (
            await session.execute(
                text(
                    "SELECT scope_kind, organization_id, org_unit_id, course_id "
                    "FROM user_role_assignments WHERE user_id = :uid"
                ),
                {"uid": str(seeded_users.hod_id)},
            )
        ).one()

    assert row.scope_kind == "org_unit"
    assert row.organization_id == seeded_users.organization_id
    assert row.org_unit_id == seeded_users.org_unit_id
    assert row.course_id is None
