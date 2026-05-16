"""JWT issue/decode using PyJWT (HS256).

Algorithm pinned, required claims enforced. Compatible with backend/ tokens via
shared HS256 + secret_key. PyJWT provides built-in algorithm-confusion guard
(via ``algorithms=[...]`` whitelist) and standard exception types.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt

from abridgeai.core.config import get_settings

ALGORITHM = "HS256"


@dataclass(frozen=True)
class TokenPayload:
    sub: UUID
    sid: UUID
    exp: datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


def create_access_token(
    *,
    user_id: UUID,
    session_id: UUID,
    expires_delta: timedelta | None = None,
) -> str:
    settings = get_settings()
    now = utcnow()
    expires_at = now + (
        expires_delta or timedelta(seconds=settings.access_token_ttl_seconds)
    )
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


__all__ = ["TokenPayload", "create_access_token", "decode_access_token", "utcnow"]
