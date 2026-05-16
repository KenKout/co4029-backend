"""Structured JSON logging via structlog.

Public API: :func:`get_logger`, :func:`configure_structlog`,
:func:`bind_request_context`, :func:`clear_request_context`.

Bridges Python's stdlib logging through the same processor chain so libraries
(httpx, sqlalchemy, redis-py, asyncio) emit JSON in production. Mode switches
between JSON (prod, machine-parseable for Loki/Datadog/CloudWatch) and
Console (dev, colored human-readable) via the LOG_FORMAT env var.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars
from structlog.types import Processor

from abridgeai.core.config import get_settings

_CONFIGURED = False

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "asyncio",
    "urllib3",
    "boto3",
    "botocore",
    "s3transfer",
)


def _shared_processors() -> list[Processor]:
    """Processors common to structlog and the stdlib bridge.

    Order matters: contextvars first (so request_id/user_id are in the event
    dict before any later processor inspects them), then level, exc info, then
    timestamp. The renderer runs LAST and lives outside this list — see
    :func:`configure_structlog`.
    """
    return [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]


def configure_structlog() -> None:
    """Wire structlog + stdlib logging once per process.

    Idempotent: subsequent calls short-circuit so tests calling this multiple
    times (or modules importing it on first use) cannot double-bind processors,
    which would corrupt output (duplicate timestamps, two render passes).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared = _shared_processors()

    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: Processor
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:  # noqa: ANN401  -- structlog.stdlib.BoundLogger generics break under strict mypy without runtime cost
    """Return a structlog BoundLogger.

    Configures the global pipeline on first use so callers don't need to call
    :func:`configure_structlog` themselves. Pass ``__name__`` for module-scoped
    loggers (matches the stdlib idiom).
    """
    if not _CONFIGURED:
        configure_structlog()
    return structlog.stdlib.get_logger(name)


def bind_request_context(**values: object) -> None:
    """Bind request-scoped key/values onto every log record from this task.

    Wraps :func:`structlog.contextvars.bind_contextvars` so HTTP/worker
    middleware (T0.23) has a stable import path. Typical keys: ``request_id``,
    ``user_id``, ``session_id``. Values flow through ``merge_contextvars``
    automatically — no need to pass them at the call site of ``log.info(...)``.
    """
    bind_contextvars(**values)


def clear_request_context() -> None:
    """Clear all bound contextvars. Call at request-end to avoid bleed across requests."""
    clear_contextvars()


def _reset_for_tests() -> None:
    """Reset module state so tests can reconfigure between cases.

    NOT a public API. Used only by ``tests/unit/test_logging.py``.
    """
    global _CONFIGURED
    _CONFIGURED = False
    structlog.reset_defaults()
    clear_contextvars()
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
