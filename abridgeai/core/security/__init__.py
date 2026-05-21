"""JWT issue/decode + ``CurrentUser`` resolution.

Algorithm pinned, required claims enforced. Compatible with backend/ tokens via
shared HS256 + secret_key. PyJWT provides built-in algorithm-confusion guard
(via ``algorithms=[...]`` whitelist) and standard exception types.

The :class:`CurrentUser` dataclass and :func:`get_current_user` FastAPI
dependency are the canonical request-scoped identity primitives consumed by
``features.access_control.policies`` (T1.11) and downstream feature routers.

Architectural note: this module is in ``core/``, which sits *under* every
feature. To preserve the import-linter ``features-are-independent`` contract
the module never imports from ``features.*``. The auth lookup uses raw SQL
against ``auth_sessions`` + ``users``; effective permissions are NOT loaded
here -- ``policies.require_*`` factories load them on demand.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.audit.context import current_actor_var
from abridgeai.core.config import Settings, get_settings
from abridgeai.core.db import get_db

ALGORITHM = "HS256"

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class TokenPayload:
    sub: UUID
    sid: UUID
    exp: datetime


@dataclass(frozen=True)
class CurrentUser:
    """Request-scoped principal resolved from a bearer token."""

    user_id: UUID
    session_id: UUID
    permissions: frozenset[str] = field(default_factory=frozenset)

    def has_permission(self, perm_code: str) -> bool:
        return perm_code in self.permissions

    def has_any_permission(self, *perm_codes: str) -> bool:
        return any(p in self.permissions for p in perm_codes)

    def with_permissions(self, perms: frozenset[str]) -> CurrentUser:
        return CurrentUser(user_id=self.user_id, session_id=self.session_id, permissions=perms)


def utcnow() -> datetime:
    return datetime.now(UTC)


def generate_token() -> str:
    return secrets.token_urlsafe(48)


def hash_secret(value: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    digest = hmac.new(settings.jwt_secret_key.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"sha256:{digest}"


def verify_secret(value: str, hashed_value: str, settings: Settings | None = None) -> bool:
    return hmac.compare_digest(hash_secret(value, settings), hashed_value)


def _fernet(settings: Settings | None = None) -> Fernet:
    settings = settings or get_settings()
    key = base64.urlsafe_b64encode(hashlib.sha256(settings.jwt_secret_key.encode()).digest())
    return Fernet(key)


def encrypt_secret(value: str, settings: Settings | None = None) -> str:
    encrypted: bytes = _fernet(settings).encrypt(value.encode())
    return encrypted.decode()


def decrypt_secret(value: str, settings: Settings | None = None) -> str:
    try:
        decrypted: bytes = _fernet(settings).decrypt(value.encode())
    except InvalidToken as exc:
        raise ValueError("Invalid encrypted secret") from exc
    return decrypted.decode()


def create_access_token(
    *,
    user_id: UUID,
    session_id: UUID,
    expires_delta: timedelta | None = None,
) -> str:
    settings = get_settings()
    now = utcnow()
    expires_at = now + (expires_delta or timedelta(seconds=settings.access_token_ttl_seconds))
    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenPayload:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[ALGORITHM],
            options={"require": ["sub", "sid", "exp"]},
        )
        return TokenPayload(
            sub=UUID(payload["sub"]),
            sid=UUID(payload["sid"]),
            exp=datetime.fromtimestamp(payload["exp"], UTC),
        )
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid access token") from exc


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


_RESOLVE_PRINCIPAL_SQL = text(
    """
    SELECT u.id           AS user_id,
           u.status       AS user_status,
           s.id           AS session_id,
           s.revoked_at   AS revoked_at,
           s.expires_at   AS expires_at,
           s.mfa_verified_at AS mfa_verified_at,
           EXISTS (
               SELECT 1 FROM mfa_factors f
               WHERE f.user_id = u.id
                 AND f.verified_at IS NOT NULL
                 AND f.disabled_at IS NULL
           ) AS user_has_verified_mfa
    FROM users u
    JOIN auth_sessions s ON s.user_id = u.id
    WHERE s.id = :session_id
      AND u.id = :user_id
    """
)


def _mfa_required() -> HTTPException:
    """403 raised when the caller's session has not completed MFA but
    the user has at least one verified MFA factor on record. Frontend
    distinguishes this from generic 403 via ``error: 'mfa_required'``
    and redirects to ``/login/mfa``.
    """
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "mfa_required"},
    )


async def _resolve_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
    db: AsyncSession,
) -> tuple[CurrentUser, bool, bool]:
    """Shared core of bearer-token resolution.

    Returns a tuple ``(current_user, mfa_pending, has_verified_mfa)``:

    - ``mfa_pending`` is ``True`` when the user owns a verified MFA
      factor but the current session has not yet been marked
      ``mfa_verified_at``.
    - ``has_verified_mfa`` mirrors the underlying flag for callers that
      want to gate an action ("reveal recovery codes only after MFA").

    Raises ``HTTPException(401)`` for invalid tokens / dead sessions
    and ``HTTPException(403)`` for inactive users. The caller decides
    whether to additionally raise ``mfa_required``.
    """
    if credentials is None:
        raise _unauthorized()

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise _unauthorized("Invalid or expired token") from exc

    result = await db.execute(
        _RESOLVE_PRINCIPAL_SQL,
        {"user_id": payload.sub, "session_id": payload.sid},
    )
    row = result.mappings().one_or_none()
    if row is None:
        raise _unauthorized("Session not found")

    if row["revoked_at"] is not None or (
        row["expires_at"] is not None and row["expires_at"] <= utcnow()
    ):
        raise _unauthorized("Session expired")
    if row["user_status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "user_inactive"},
        )

    has_verified_mfa = bool(row["user_has_verified_mfa"])
    mfa_pending = has_verified_mfa and row["mfa_verified_at"] is None

    current = CurrentUser(user_id=row["user_id"], session_id=row["session_id"])
    request.state.user = current
    # Bind the actor for the rest of this request's contextvars.Context.
    # FastAPI/Starlette runs each request in its own asyncio.Task, which holds
    # an isolated copy of the context, so the bind dies when the handler
    # returns. The companion ``after_begin`` listener in ``core/db/__init__``
    # propagates this UUID to PostgreSQL via ``set_config('app.actor_id',
    # ..., true)`` on every transaction begin.
    current_actor_var.set(current.user_id)
    return current, mfa_pending, has_verified_mfa


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> CurrentUser:
    """Resolve the authenticated principal from a bearer token.

    Validates the JWT, confirms the referenced ``auth_sessions`` row is live
    (not revoked, not expired) and the ``users`` row is active. Raises
    ``HTTPException(401)`` on missing / invalid tokens or revoked sessions
    and ``HTTPException(403)`` on inactive users.

    MFA gate: when the user has at least one verified MFA factor and the
    current session has NOT yet completed MFA verification, this raises
    ``HTTPException(403, {"error": "mfa_required"})``. The frontend
    intercepts that response and redirects to ``/login/mfa``. Endpoints
    that legitimately need to run before MFA (challenge / verify /
    logout) depend on :func:`get_current_user_pre_mfa` instead.

    The returned :class:`CurrentUser` carries an EMPTY permission set --
    permission resolution lives in ``features.access_control.policies`` so
    this module avoids a cross-feature import and the import-linter
    ``features-are-independent`` contract stays green.
    """
    current, mfa_pending, _has_verified_mfa = await _resolve_principal(request, credentials, db)
    if mfa_pending:
        raise _mfa_required()
    return current


async def get_current_user_pre_mfa(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
) -> CurrentUser:
    """Same as :func:`get_current_user` but DOES NOT enforce the MFA gate.

    Intended for the endpoints that must run while the session still has
    ``mfa_verified_at IS NULL`` so the user can complete MFA — namely
    ``POST /auth/mfa/challenge``, ``POST /auth/mfa/verify`` and
    ``POST /auth/logout``. Every other endpoint should use
    :func:`get_current_user`.
    """
    current, _mfa_pending, _has_verified_mfa = await _resolve_principal(request, credentials, db)
    return current


__all__ = [
    "ALGORITHM",
    "CurrentUser",
    "TokenPayload",
    "create_access_token",
    "decode_access_token",
    "decrypt_secret",
    "encrypt_secret",
    "generate_token",
    "get_current_user",
    "get_current_user_pre_mfa",
    "hash_secret",
    "utcnow",
    "verify_secret",
]
