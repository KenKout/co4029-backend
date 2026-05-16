"""Audit context — actor identity carrier.

`current_actor_var` is a `contextvars.ContextVar` that holds the UUID of
the user (or system actor) responsible for the current unit of work.

It is set per-request by `AuditContextMiddleware` (HTTP path) and per-task
by `set_worker_actor` (ARQ worker path). Both runtimes use asyncio Tasks,
which inherit the parent context — so any code path reachable from those
entry points (services, repositories, SQLAlchemy listeners) can read the
actor without explicit wiring.

When unset (e.g. system migrations, scheduled jobs without an attributed
caller), `get_current_actor()` returns None and the audit listener leaves
`created_by` / `updated_by` NULL.
"""

from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID

current_actor_var: ContextVar[UUID | None] = ContextVar(
    "current_actor", default=None
)


def get_current_actor() -> UUID | None:
    """Return the actor UUID for the current async context, or None."""
    return current_actor_var.get()


__all__ = ["current_actor_var", "get_current_actor"]
