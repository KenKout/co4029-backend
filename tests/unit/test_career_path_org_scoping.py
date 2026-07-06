"""Unit tests for FR-2.6 org scoping on management career-path reads (phase-07).

`list_path_roster_progress` and `get_path_readiness_overview` expose
student emails; the permission deps are global-scope, so the router gate
`_ensure_caller_in_path_org` must reject cross-org callers with 404 (no
existence leak) while letting org members and platform admins through.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from abridgeai.features.career_paths.routers import authoring as router

_PATH = uuid.uuid4()
_ORG = uuid.uuid4()
_USER = SimpleNamespace(user_id=uuid.uuid4())


def _patched(member: bool, admin: bool) -> list:
    perms = (
        [SimpleNamespace(code="system.administer")]
        if admin
        else [SimpleNamespace(code="course.enrollment.read")]
    )
    return [
        patch.object(
            router.authoring_service,
            "get_career_path",
            new=AsyncMock(return_value=SimpleNamespace(id=_PATH, organization_id=_ORG)),
        ),
        patch.object(
            router.access_control_api,
            "is_user_member_of_org",
            new=AsyncMock(return_value=member),
        ),
        patch.object(
            router.access_control_api,
            "get_active_permissions",
            new=AsyncMock(return_value=perms),
        ),
    ]


class TestEnsureCallerInPathOrg:
    async def test_org_member_passes(self) -> None:
        p = _patched(member=True, admin=False)
        with p[0], p[1], p[2]:
            await router._ensure_caller_in_path_org(AsyncMock(), _USER, _PATH)

    async def test_platform_admin_passes(self) -> None:
        p = _patched(member=False, admin=True)
        with p[0], p[1], p[2]:
            await router._ensure_caller_in_path_org(AsyncMock(), _USER, _PATH)

    async def test_cross_org_manager_gets_404(self) -> None:
        p = _patched(member=False, admin=False)
        with p[0], p[1], p[2], pytest.raises(HTTPException) as exc_info:
            await router._ensure_caller_in_path_org(AsyncMock(), _USER, _PATH)
        assert exc_info.value.status_code == 404


class TestGateWiring:
    async def test_roster_route_invokes_gate(self) -> None:
        gate = AsyncMock()
        with (
            patch.object(router, "_ensure_caller_in_path_org", new=gate),
            patch.object(
                router.enrollment_service, "get_roster_progress", new=AsyncMock(return_value=[])
            ),
        ):
            await router.list_path_roster_progress(_PATH, _USER, AsyncMock())
        gate.assert_awaited_once()

    async def test_readiness_route_invokes_gate(self) -> None:
        gate = AsyncMock()
        with (
            patch.object(router, "_ensure_caller_in_path_org", new=gate),
            patch.object(
                router.readiness_service,
                "get_path_readiness_overview",
                new=AsyncMock(return_value=SimpleNamespace()),
            ),
        ):
            await router.get_path_readiness_overview(_PATH, _USER, AsyncMock())
        gate.assert_awaited_once()
