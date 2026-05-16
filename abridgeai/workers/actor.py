"""ARQ worker — actor binding helper.

ARQ task convention (Reconciliation §B6, plan T0.8): every task signature is
`async def task(ctx, actor_id: UUID, ...)`. The task body's first action is
to call `set_worker_actor(actor_id)` to install the actor into the audit
context, so any DB writes made during the task automatically populate
`created_by` / `updated_by`.

Each ARQ task runs in its own asyncio Task, which carries its own
`contextvars.Context` snapshot — no manual reset is required between
tasks. The returned `Token` is provided for callers that explicitly want
to scope the binding (e.g. nested or test contexts).
"""

from __future__ import annotations

from contextvars import Token
from uuid import UUID

from abridgeai.core.audit import current_actor_var


def set_worker_actor(actor_id: UUID | None) -> Token:
    """Bind the given actor to the current async context.

    Returns the token for optional `current_actor_var.reset(token)` use.
    """
    return current_actor_var.set(actor_id)


__all__ = ["set_worker_actor"]
