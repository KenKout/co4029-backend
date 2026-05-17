"""HTTP audit log middleware (T0.23).

Persists every non-skip request to ``http_audit_log`` (migration 0011) with
header/query redaction and a propagated ``X-Request-ID``. Best-effort: a DB
failure NEVER crashes the request, the structlog event always fires.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any, Final

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from abridgeai.core.observability.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)

_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

_SKIP_PATHS: Final[frozenset[str]] = frozenset(
    {"/healthz", "/metrics", "/docs", "/redoc", "/openapi.json"}
)

_REDACT_QUERY_KEYS: Final[frozenset[str]] = frozenset(
    {"code", "token", "secret", "api_key", "apikey", "password", "authorization"}
)

_REDACT_HEADER_KEYS: Final[frozenset[str]] = frozenset(
    {"authorization", "cookie", "x-api-key", "x-csrf-token"}
)

_HEADER_ALLOW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "content-type",
        "content-length",
        "accept",
        "accept-language",
        "host",
        "origin",
        "referer",
        "x-forwarded-for",
        "x-real-ip",
    }
)

_REDACTED: Final[str] = "[REDACTED]"

_MUTATION_METHODS: Final[frozenset[str]] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_INSERT_SQL = text(
    """
    INSERT INTO http_audit_log (
        request_id, method, path, query_params, headers_meta,
        status_code, latency_ms, user_id, session_id, ip_address,
        user_agent, path_params, body_size_bytes
    ) VALUES (
        :request_id, :method, :path, CAST(:query_params AS jsonb),
        CAST(:headers_meta AS jsonb),
        :status_code, :latency_ms, :user_id, :session_id, :ip_address,
        :user_agent, CAST(:path_params AS jsonb), :body_size_bytes
    )
    """
)


def _redact_query_params(items: list[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw_k, v in items:
        k = raw_k.lower()
        value = _REDACTED if k in _REDACT_QUERY_KEYS else v
        out.setdefault(raw_k, []).append(value)
    return out


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_k, v in headers.items():
        k = raw_k.lower()
        if k in _REDACT_HEADER_KEYS:
            out[k] = _REDACTED
        elif k in _HEADER_ALLOW_KEYS:
            out[k] = v
    return out


def _client_ip(request: Request) -> str | None:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip() or None
    if request.client is not None:
        return request.client.host
    return None


def _user_id(request: Request) -> uuid.UUID | None:
    user = getattr(request.state, "user", None)
    if user is None:
        return None
    candidate = getattr(user, "id", None)
    return candidate if isinstance(candidate, uuid.UUID) else None


def _session_id(request: Request) -> uuid.UUID | None:
    session = getattr(request.state, "session", None)
    if session is None:
        return None
    candidate = getattr(session, "id", None)
    return candidate if isinstance(candidate, uuid.UUID) else None


def _body_size(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if raw is None:
        return None
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return None
    return size if size >= 0 else None


def _path_params(request: Request) -> dict[str, str] | None:
    params = request.scope.get("path_params") or {}
    if not params:
        return None
    return {k: str(v) for k, v in params.items()}


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else value[:limit]


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Persist every HTTP request to ``http_audit_log`` (T0.23)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        sessionmaker: async_sessionmaker[Any] | None = None,
    ) -> None:
        super().__init__(app)
        self._sessionmaker = sessionmaker
        self._log = get_logger(__name__)

    def _resolve_sessionmaker(self) -> async_sessionmaker[Any] | None:
        if self._sessionmaker is not None:
            return self._sessionmaker
        try:
            from abridgeai.core.db import get_sessionmaker
        except ImportError:
            return None
        try:
            return get_sessionmaker()
        except Exception:
            return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = uuid.uuid4()
        bind_request_context(request_id=str(request_id))
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            pass

        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers[_REQUEST_ID_HEADER] = str(request_id)

        if request.url.path in _SKIP_PATHS and request.method == "GET":
            clear_request_context()
            return response

        method = request.method.upper()
        path = _truncate(request.url.path, 2000) or ""
        query_params = _redact_query_params(list(request.query_params.multi_items()))
        headers_meta = _redact_headers(request.headers)
        user_agent = _truncate(request.headers.get("user-agent"), 500)
        ip_address = _client_ip(request)
        user_id = _user_id(request)
        session_id = _session_id(request)
        is_mutation = method in _MUTATION_METHODS
        path_params = _path_params(request) if is_mutation else None
        body_size = _body_size(request) if is_mutation else None

        log_payload: dict[str, Any] = {
            "request_id": str(request_id),
            "method": method,
            "path": path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "user_id": str(user_id) if user_id else None,
            "ip_address": ip_address,
        }

        await self._persist(
            request_id=request_id,
            method=method,
            path=path,
            query_params=query_params,
            headers_meta=headers_meta,
            status_code=response.status_code,
            latency_ms=latency_ms,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            path_params=path_params,
            body_size_bytes=body_size,
            log_payload=log_payload,
        )

        self._log.info("http_request", **log_payload)
        clear_request_context()
        return response

    async def _persist(
        self,
        *,
        request_id: uuid.UUID,
        method: str,
        path: str,
        query_params: dict[str, list[str]],
        headers_meta: dict[str, str],
        status_code: int,
        latency_ms: int,
        user_id: uuid.UUID | None,
        session_id: uuid.UUID | None,
        ip_address: str | None,
        user_agent: str | None,
        path_params: dict[str, str] | None,
        body_size_bytes: int | None,
        log_payload: dict[str, Any],
    ) -> None:
        sessionmaker = self._resolve_sessionmaker()
        if sessionmaker is None:
            self._log.warning("audit_log_persist_unavailable", **log_payload)
            return

        import json

        params: dict[str, Any] = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "query_params": json.dumps(query_params),
            "headers_meta": json.dumps(headers_meta),
            "status_code": status_code,
            "latency_ms": latency_ms,
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "path_params": json.dumps(path_params) if path_params is not None else None,
            "body_size_bytes": body_size_bytes,
        }

        try:
            async with sessionmaker() as session:
                await session.execute(_INSERT_SQL, params)
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            self._log.warning(
                "audit_log_persist_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                **log_payload,
            )


def register_audit_log_middleware(app: FastAPI) -> None:
    """Register :class:`AuditLogMiddleware` on a FastAPI/Starlette app."""
    app.add_middleware(AuditLogMiddleware)


__all__ = ["AuditLogMiddleware", "register_audit_log_middleware"]
