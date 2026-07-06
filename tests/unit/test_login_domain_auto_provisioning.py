"""Unit tests for OAuth domain auto-provisioning (FR-2.7/FR-2.9, phase-05).

Covers the ``_auto_provision_user`` hook in the Google callback. The
access_control reach is dependency-injected (identity services must stay
feature-local per the source-grep guard in test_identity_services.py),
so tests pass the callables directly — no DB, no cross-feature patches.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from abridgeai.core.exceptions import ForbiddenError
from abridgeai.features.identity.services import login as login_service

_ORG = uuid.uuid4()
_SVC = "abridgeai.features.identity.services.login"


def _google_profile(
    email: str = "new.student@uni.edu", *, email_verified: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        subject="google-sub-123",
        email=email,
        given_name="New",
        family_name="Student",
        display_name="New Student",
        email_verified=email_verified,
    )


class TestAutoProvisionUser:
    async def test_unwired_callables_return_none(self) -> None:
        result = await login_service._auto_provision_user(
            AsyncMock(),
            email="someone@uni.edu",
            resolve_auto_provision_org=None,
            grant_default_access=None,
        )
        assert result is None

    async def test_no_matching_domain_returns_none(self) -> None:
        resolve = AsyncMock(return_value=None)
        result = await login_service._auto_provision_user(
            AsyncMock(),
            email="someone@unknown.example",
            resolve_auto_provision_org=resolve,
            grant_default_access=AsyncMock(),
        )
        assert result is None
        assert resolve.await_args.args[1] == "unknown.example"

    async def test_matching_domain_creates_user_with_student_access(self) -> None:
        db = AsyncMock()
        resolve = AsyncMock(return_value=_ORG)
        grant = AsyncMock()
        with patch(f"{_SVC}.flush_or_conflict", new=AsyncMock()):
            user = await login_service._auto_provision_user(
                db,
                email="Case.Mixed@Uni.EDU",
                resolve_auto_provision_org=resolve,
                grant_default_access=grant,
            )
        assert user is not None
        assert user.primary_email == "case.mixed@uni.edu"
        assert user.status == "active"
        db.add.assert_called_once()
        grant.assert_awaited_once()
        assert grant.await_args.args[2] == _ORG
        assert resolve.await_args.args[1] == "uni.edu"


class TestGoogleCallbackProvisioningPath:
    async def test_unknown_email_without_provisioner_still_403(self) -> None:
        with (
            patch(
                f"{_SVC}.fetch_google_profile",
                new=AsyncMock(return_value=_google_profile()),
            ),
            patch.object(
                login_service.user_queries,
                "get_identity_by_provider_subject",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                login_service.user_queries,
                "get_user_by_email",
                new=AsyncMock(return_value=None),
            ),
            pytest.raises(ForbiddenError, match="not registered"),
        ):
            await login_service.handle_google_callback(AsyncMock(), code="code")

    async def test_unknown_email_unmatched_domain_403(self) -> None:
        with (
            patch(
                f"{_SVC}.fetch_google_profile",
                new=AsyncMock(return_value=_google_profile()),
            ),
            patch.object(
                login_service.user_queries,
                "get_identity_by_provider_subject",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                login_service.user_queries,
                "get_user_by_email",
                new=AsyncMock(return_value=None),
            ),
            pytest.raises(ForbiddenError, match="not registered"),
        ):
            await login_service.handle_google_callback(
                AsyncMock(),
                code="code",
                resolve_auto_provision_org=AsyncMock(return_value=None),
                grant_default_access=AsyncMock(),
            )

    async def test_unverified_email_never_provisions(self) -> None:
        """FR-2.7 gate: unverified OIDC email → invite-only 403, hook untouched."""
        hook = AsyncMock()
        with (
            patch(
                f"{_SVC}.fetch_google_profile",
                new=AsyncMock(return_value=_google_profile(email_verified=False)),
            ),
            patch.object(
                login_service.user_queries,
                "get_identity_by_provider_subject",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                login_service.user_queries,
                "get_user_by_email",
                new=AsyncMock(return_value=None),
            ),
            patch(f"{_SVC}._auto_provision_user", new=hook),
            pytest.raises(ForbiddenError, match="not registered"),
        ):
            await login_service.handle_google_callback(
                AsyncMock(),
                code="code",
                resolve_auto_provision_org=AsyncMock(return_value=_ORG),
                grant_default_access=AsyncMock(),
            )
        hook.assert_not_awaited()

    async def test_unknown_email_with_domain_provisions_and_issues_tokens(self) -> None:
        provisioned = SimpleNamespace(
            id=uuid.uuid4(), status="active", last_login_at=None, primary_email="a@uni.edu"
        )
        issue = AsyncMock(return_value="TOKENS")
        with (
            patch(
                f"{_SVC}.fetch_google_profile",
                new=AsyncMock(return_value=_google_profile()),
            ),
            patch.object(
                login_service.user_queries,
                "get_identity_by_provider_subject",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                login_service.user_queries,
                "get_user_by_email",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                login_service.user_queries, "get_profile", new=AsyncMock(return_value=None)
            ),
            patch(f"{_SVC}._auto_provision_user", new=AsyncMock(return_value=provisioned)),
            patch(f"{_SVC}._issue_tokens", new=issue),
        ):
            result = await login_service.handle_google_callback(AsyncMock(), code="code")
        assert result == "TOKENS"
        # Provisioned user got a profile + google identity + last_login stamp.
        assert provisioned.last_login_at is not None
        issue.assert_awaited_once()


class TestRouterWiring:
    async def test_callback_endpoint_injects_provisioners(self) -> None:
        from abridgeai.features.identity.routers import auth as auth_router

        svc = AsyncMock(return_value="TOKENS")
        request = SimpleNamespace(client=None, headers={})
        with patch.object(auth_router.login_service, "handle_google_callback", svc):
            await auth_router.google_callback(request, AsyncMock(), code="code")
        kwargs = svc.await_args.kwargs
        assert kwargs["resolve_auto_provision_org"] is auth_router._resolve_auto_provision_org
        assert kwargs["grant_default_access"] is auth_router._grant_default_access
