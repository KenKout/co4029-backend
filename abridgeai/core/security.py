"""Pure-stdlib HMAC-SHA256 JWT, byte-compatible with backend/app/core/security.py."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from abridgeai.core.config import Settings, get_settings


@dataclass(frozen=True)
class TokenPayload:
    sub: UUID
    sid: UUID
    exp: datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


def _signing_key(settings: Settings | None = None) -> bytes:
    settings = settings or get_settings()
    return settings.jwt_secret_key.encode()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}".encode())


def create_access_token(
    *,
    user_id: UUID,
    session_id: UUID,
    expires_delta: timedelta | None = None,
) -> str:
    settings = get_settings()
    expires_at = utcnow() + (
        expires_delta or timedelta(seconds=settings.access_token_ttl_seconds)
    )
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "exp": int(expires_at.timestamp()),
    }
    encoded_header = _b64encode(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}"
    signature = hmac.new(
        _signing_key(settings), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64encode(signature)}"


def decode_access_token(token: str) -> TokenPayload:
    settings = get_settings()
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        header = json.loads(_b64decode(encoded_header))
        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            raise ValueError("Unsupported token header")
        signing_input = f"{encoded_header}.{encoded_payload}"
        expected_signature = hmac.new(
            _signing_key(settings), signing_input.encode(), hashlib.sha256
        ).digest()
        provided_signature = _b64decode(encoded_signature)
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise ValueError("Invalid token signature")
        payload = json.loads(_b64decode(encoded_payload))
        expires_at = datetime.fromtimestamp(payload["exp"], UTC)
        if expires_at <= utcnow():
            raise ValueError("Token expired")
        return TokenPayload(
            sub=UUID(payload["sub"]),
            sid=UUID(payload["sid"]),
            exp=expires_at,
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid access token") from exc


__all__ = ["TokenPayload", "create_access_token", "decode_access_token", "utcnow"]
