"""System-actor scope for worker / cron / batch contexts (T6).

Provides ``system_actor_scope()`` -- an async context manager that binds
``current_actor_var`` to the canonical migration-seeded system user UUID
for the duration of the ``async with`` block.

Use this in worker / cron / batch entry points that have no HTTP request
actor available. The migration-0004 system user
(``00000000-0000-0000-0000-000000000001``) is stable across the suite,
so audit FKs into ``users`` always resolve.

Underscore-prefixed (private) for now -- not re-exported from
``abridgeai.core.audit.__init__``. Existing workers continue to use
:func:`abridgeai.workers.actor.set_worker_actor`; this module is the
recommended pattern for *future* worker code and may be promoted to the
public API once callers migrate.

Example
-------
    async def my_cron_task() -> None:
        async with system_actor_scope():
            ...  # any DB writes here get system actor stamped on audit cols
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from abridgeai.core.audit.context import current_actor_var

SYSTEM_ACTOR_ID = UUID("00000000-0000-0000-0000-000000000001")
"""Migration-0004 system user UUID. Stable across the suite."""


@asynccontextmanager
async def system_actor_scope() -> AsyncIterator[None]:
    """Set ``current_actor_var`` to ``SYSTEM_ACTOR_ID`` for the scope.

    Used by worker / cron / batch contexts that have no HTTP request
    actor. Restores the prior actor on exit (``None`` if previously
    unset) via :meth:`contextvars.ContextVar.reset` so concurrent or
    nested scopes never leak identity.
    """
    token = current_actor_var.set(SYSTEM_ACTOR_ID)
    try:
        yield
    finally:
        current_actor_var.reset(token)


__all__ = ["SYSTEM_ACTOR_ID", "system_actor_scope"]
