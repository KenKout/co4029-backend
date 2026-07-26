"""Org scoping on the career-path management surface.

The permission dependencies on this router are flat: ``load_user_permissions``
resolves role assignments without regard to ``scope_kind``, so a role granted
to a manager inside org B yields the same codes as a global grant. Every route
that resolves a path by id must therefore run ``_ensure_caller_in_path_org``
itself, or a manager in any org can read and mutate another org's paths.

The wiring test below is the one that matters: the gate existed and was
correct long before it was applied to more than two of the twelve routes.
"""

from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from abridgeai.features.access_control import policies
from abridgeai.features.career_paths.routers import authoring as router

_PATH = uuid.uuid4()
_ORG = uuid.uuid4()


def _user(*, admin: bool = False) -> SimpleNamespace:
    codes = {"system.administer"} if admin else {"course.update"}
    return SimpleNamespace(
        user_id=uuid.uuid4(),
        permissions=frozenset(codes),
        has_permission=codes.__contains__,
    )


def _patched(granted_here: set[str] | None) -> list:
    """``granted_here`` is what the caller holds FOR the path's organization."""
    return [
        patch.object(
            router.authoring_service,
            "get_career_path",
            new=AsyncMock(return_value=SimpleNamespace(id=_PATH, organization_id=_ORG)),
        ),
        patch.object(
            policies,
            "load_org_permissions",
            new=AsyncMock(return_value=granted_here or set()),
        ),
    ]


class TestEnsureCallerInPathOrg:
    async def test_permission_granted_for_this_org_passes(self) -> None:
        get_path, granted = _patched({"course.update"})
        with get_path, granted:
            await router._ensure_caller_in_path_org(AsyncMock(), _user(), _PATH)

    async def test_platform_admin_passes(self) -> None:
        get_path, granted = _patched(set())
        with get_path, granted:
            await router._ensure_caller_in_path_org(AsyncMock(), _user(admin=True), _PATH)

    async def test_permission_held_only_in_another_org_gets_404(self) -> None:
        """The case bare membership could not catch.

        The caller genuinely holds ``course.update`` — the route dependency
        already let them through on it — but it was granted in a different
        organization, so nothing is granted *here*.
        """
        get_path, granted = _patched(set())
        with get_path, granted, pytest.raises(HTTPException) as exc_info:
            await router._ensure_caller_in_path_org(AsyncMock(), _user(), _PATH)
        # 404 not 403: a 403 would confirm the path exists, turning the id
        # endpoint into a cross-tenant existence oracle.
        assert exc_info.value.status_code == 404

    async def test_a_different_permission_here_does_not_authorise(self) -> None:
        """Holding *some* permission in the org is not holding the right one."""
        get_path, granted = _patched({"progress.read.cohort"})
        with get_path, granted, pytest.raises(HTTPException) as exc_info:
            await router._ensure_caller_in_path_org(AsyncMock(), _user(), _PATH)
        assert exc_info.value.status_code == 404


class TestEveryByIdRouteIsGated:
    """No route may resolve a ``career_path_id`` without calling the gate.

    Asserted over the source of each handler rather than by exercising it, so
    a new endpoint added later fails here even if nobody writes a test for it.
    """

    def test_all_career_path_id_handlers_call_the_gate(self) -> None:
        ungated: list[str] = []
        for name, fn in vars(router).items():
            if not inspect.isfunction(fn) or name.startswith("_"):
                continue
            signature = inspect.signature(fn)
            if "career_path_id" not in signature.parameters:
                continue
            source = inspect.getsource(fn)
            if "_ensure_caller_in_path_org" not in source:
                ungated.append(name)
        assert ungated == [], (
            f"career-path routes resolving an id without an org check: {ungated}. "
            "Add `await _ensure_caller_in_path_org(db, current_user, career_path_id)` "
            "inside the handler's try block."
        )

    def test_the_check_covers_more_than_the_two_original_reads(self) -> None:
        """Regression guard on the bug itself.

        The gate was written for the two roster/readiness reads and left
        unwired everywhere else; a count well above two is what says the rest
        of the surface is covered too.
        """
        gated = sum(
            1
            for name, fn in vars(router).items()
            if inspect.isfunction(fn)
            and not name.startswith("_")
            and "career_path_id" in inspect.signature(fn).parameters
            and "_ensure_caller_in_path_org" in inspect.getsource(fn)
        )
        assert gated >= 12, f"only {gated} career-path routes are gated"


class TestListHonoursOrgOverride:
    async def test_explicit_organization_id_is_scope_checked(self) -> None:
        """``?organization_id=`` must not be a cross-tenant read.

        The parameter was honoured unconditionally "so platform admins can
        list any org", which let any manager enumerate another org's paths.
        """
        other_org = uuid.uuid4()
        with (
            patch.object(policies, "load_org_permissions", new=AsyncMock(return_value=set())),
            patch.object(
                router.authoring_service,
                "list_career_paths_for_org",
                new=AsyncMock(return_value=[]),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await router.list_career_paths(
                _user(), AsyncMock(), organization_id=other_org
            )
        assert exc_info.value.status_code == 404

    async def test_admin_may_still_pass_any_org(self) -> None:
        listing = AsyncMock(return_value=[])
        with (
            patch.object(policies, "load_org_permissions", new=AsyncMock(return_value=set())),
            patch.object(router.authoring_service, "list_career_paths_for_org", new=listing),
        ):
            await router.list_career_paths(
                _user(admin=True), AsyncMock(), organization_id=uuid.uuid4()
            )
        listing.assert_awaited_once()


class TestGateWiring:
    async def test_roster_route_invokes_gate(self) -> None:
        gate = AsyncMock()
        with (
            patch.object(router, "_ensure_caller_in_path_org", new=gate),
            patch.object(
                router.enrollment_service, "get_roster_progress", new=AsyncMock(return_value=[])
            ),
        ):
            await router.list_path_roster_progress(_PATH, _user(), AsyncMock())
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
            await router.get_path_readiness_overview(_PATH, _user(), AsyncMock())
        gate.assert_awaited_once()
