"""Identity OAuth router — Google login, callback, refresh, logout.

Auth-correct from day 1: OAuth-only (no credential endpoints), no
permission-bypass flag. Routers MUST go through ``services.*``; direct
``queries.*`` access would break the import-linter ``Routers do not call
queries directly`` contract (T0.4 #2).

Service-commit discipline: T1.7 services (``handle_google_callback``,
``refresh_tokens``, ``logout``) commit their own transactions, so this router
is a thin handler that translates application exceptions to HTTP responses.
"""

from __future__ import annotations

import secrets
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import AppError, ForbiddenError, UnauthorizedError
from abridgeai.core.security import (
    CurrentUser,
    get_optional_current_user_pre_mfa,
)
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.identity.schemas import (
    GoogleLoginResponse,
    LogoutRequest,
    RefreshTokenRequest,
    TokenResponse,
)
from abridgeai.features.identity.services import login as login_service
from abridgeai.features.identity.services import session as session_service
from abridgeai.infrastructure.google_oauth import build_authorization_url

router = APIRouter(prefix="/auth", tags=["auth"])


# FR-2.7/FR-2.9 — domain auto-provisioning adapters. The identity SERVICE
# layer must stay feature-local (source-grep guard in
# tests/unit/test_identity_services.py), so the cross-feature reach into
# access_control happens here and is injected into the login service.
async def _resolve_auto_provision_org(db: AsyncSession, email_domain: str) -> UUID | None:
    return await access_control_api.find_auto_provision_org_id(db, email_domain=email_domain)


async def _grant_default_access(db: AsyncSession, user_id: UUID, org_id: UUID) -> None:
    await access_control_api.grant_default_student_access(
        db, user_id=user_id, organization_id=org_id
    )


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.get("/google/login", response_model=GoogleLoginResponse)
async def google_login() -> GoogleLoginResponse:
    """Return the Google OAuth authorization URL plus an opaque ``state``.

    The SPA stores ``state`` and re-presents it on the callback so it can
    detect cross-site request forgery on the redirect leg.
    """
    state = secrets.token_urlsafe(24)
    try:
        authorization_url = build_authorization_url(state)
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return GoogleLoginResponse(authorization_url=authorization_url, state=state)


@router.get("/google/callback", response_model=TokenResponse)
async def google_callback(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    code: Annotated[str, Query(min_length=1)],
) -> TokenResponse:
    """Exchange a Google authorization code for an application token pair."""
    try:
        return await login_service.handle_google_callback(
            db,
            code=code,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            resolve_auto_provision_org=_resolve_auto_provision_org,
            grant_default_access=_grant_default_access,
        )
    except UnauthorizedError as exc:
        raise _unauthorized(str(exc)) from exc
    except ForbiddenError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "oauth_account_not_provisioned", "message": str(exc)},
        ) from exc
    except AppError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Issue a new access/refresh token pair for an active session."""
    try:
        return await session_service.refresh_tokens(db, payload.refresh_token)
    except UnauthorizedError as exc:
        raise _unauthorized(str(exc)) from exc


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: Annotated[
        CurrentUser | None, Depends(get_optional_current_user_pre_mfa)
    ],
    payload: LogoutRequest | None = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> Response:
    """Revoke the caller's session. Idempotent — never a client-visible failure.

    Two resolution paths, tried in order:

    1. Bearer token (when it still validates — includes users mid-MFA, so
       abandoning the login flow works: that is the whole point of
       ``get_optional_current_user_pre_mfa``).
    2. The body's ``refresh_token``: the client sends the token it holds, so
       sign-out still revokes the session when the ACCESS token has expired
       (the exact case where the old route 401'd and the frontend's
       refresh-then-logout dance could hang the UI on a dead socket).

    Both fail (unknown token, already revoked) → still 204: the client's goal
    — "this session must die" — is already true or unachievable, and either
    way the sign-out UX must not error.
    """
    if current_user is not None:
        await session_service.logout(db, session_id=current_user.session_id)
    elif payload is not None and payload.refresh_token:
        await session_service.logout(db, refresh_token=payload.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
