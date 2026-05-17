"""Tests for ``abridgeai.core.audit._worker_actor.system_actor_scope`` (T6)."""

from __future__ import annotations

from uuid import UUID

import pytest

from abridgeai.core.audit._worker_actor import SYSTEM_ACTOR_ID, system_actor_scope
from abridgeai.core.audit.context import current_actor_var


@pytest.mark.asyncio
async def test_scope_set_and_reset() -> None:
    assert current_actor_var.get() is None

    async with system_actor_scope():
        assert current_actor_var.get() == SYSTEM_ACTOR_ID

    assert current_actor_var.get() is None


@pytest.mark.asyncio
async def test_nested_scopes() -> None:
    custom_uuid = UUID("11111111-2222-3333-4444-555555555555")
    outer_token = current_actor_var.set(custom_uuid)
    try:
        assert current_actor_var.get() == custom_uuid

        async with system_actor_scope():
            assert current_actor_var.get() == SYSTEM_ACTOR_ID

        assert current_actor_var.get() == custom_uuid
    finally:
        current_actor_var.reset(outer_token)

    assert current_actor_var.get() is None
