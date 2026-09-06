from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.core.exceptions import ConflictError, ForbiddenError
from abridgeai.features.access_control.schemas.admin import MembershipCreate
from abridgeai.features.access_control.services import admin as service
from abridgeai.features.identity.schemas.profile import UserCreate


@pytest.mark.asyncio
async def test_platform_admin_cannot_receive_org_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    is_platform_admin = AsyncMock(return_value=True)
    reserved_lookup = AsyncMock()
    monkeypatch.setattr(service.admin_queries, "is_platform_admin", is_platform_admin)
    monkeypatch.setattr(
        service.admin_queries, "get_reserved_membership_for_user", reserved_lookup
    )

    payload = MembershipCreate(user_id=uuid4())
    with pytest.raises(ForbiddenError, match="platform_admin_cannot_join_an_organization"):
        await service.add_organization_membership(
            SimpleNamespace(), organization_id=uuid4(), payload=payload
        )

    reserved_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_left_membership_reserves_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service.admin_queries, "is_platform_admin", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        service.admin_queries,
        "get_reserved_membership_for_user",
        AsyncMock(return_value=SimpleNamespace(status="suspended")),
    )
    insert = AsyncMock()
    monkeypatch.setattr(service.admin_queries, "insert_membership", insert)

    payload = MembershipCreate(user_id=uuid4(), status="inactive")
    with pytest.raises(ConflictError, match="user_already_belongs_to_an_organization"):
        await service.add_organization_membership(
            SimpleNamespace(), organization_id=uuid4(), payload=payload
        )

    insert.assert_not_awaited()


@pytest.mark.asyncio
async def test_unreserved_regular_user_can_receive_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service.admin_queries, "is_platform_admin", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        service.admin_queries,
        "get_reserved_membership_for_user",
        AsyncMock(return_value=None),
    )
    expected = SimpleNamespace(id=uuid4())
    insert = AsyncMock(return_value=expected)
    monkeypatch.setattr(service.admin_queries, "insert_membership", insert)

    organization_id = uuid4()
    payload = MembershipCreate(user_id=uuid4(), student_code="SV-001")
    result = await service.add_organization_membership(
        SimpleNamespace(), organization_id=organization_id, payload=payload
    )

    assert result is expected
    insert.assert_awaited_once()


def test_invite_identifier_matches_selected_role() -> None:
    UserCreate(primary_email="student@example.com", student_code="SV-001")
    UserCreate(
        primary_email="teacher@example.com",
        role_code="teacher",
        employee_code="NV-001",
    )

    with pytest.raises(ValueError, match="employee_code is not valid"):
        UserCreate(primary_email="student@example.com", employee_code="NV-001")
    with pytest.raises(ValueError, match="student_code is only valid"):
        UserCreate(
            primary_email="teacher@example.com",
            role_code="teacher",
            student_code="SV-001",
        )
