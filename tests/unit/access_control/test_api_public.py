from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.access_control import policies as _policies
from abridgeai.features.access_control.api.public import (
    OrgUnitDTO,
    can_manage_course,
    get_org_unit_ancestors,
    require_any_permission,
    require_course_permission,
    require_permission,
)


def test_decorator_reexport_is_true_alias() -> None:
    assert require_permission is _policies.require_permission
    assert require_any_permission is _policies.require_any_permission
    assert require_course_permission is _policies.require_course_permission
    assert can_manage_course is _policies.can_manage_course


def test_decorator_reexport_usable_from_sibling() -> None:
    dep = require_permission("some.code")
    assert callable(dep)


@pytest.mark.asyncio
async def test_org_ancestors_single_faculty_root(test_engine: AsyncEngine) -> None:
    """Ancestor walk under the post-0094 unit shape.

    Migration 0094 flattened org units: ck_org_units_live_faculty_root makes
    every LIVE unit a top-level faculty (parent NULL) — a live 4-level tree is
    no longer legal schema. The walk still returns the unit itself and, if a
    child were ever allowed to reference a parent, would resolve ancestors;
    here the only constructible row is the root, so the walk is [self].
    """
    org_id = uuid4()
    root_id = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await session.execute(
                text(
                    "INSERT INTO organizations (id, slug, name, status, created_at, updated_at) "
                    "VALUES (:id, :slug, :name, 'active', NOW(), NOW())"
                ),
                {"id": str(org_id), "slug": f"t-{org_id.hex[:8]}", "name": "T Org"},
            )
            await session.execute(
                text(
                    "INSERT INTO org_units "
                    "(id, organization_id, parent_unit_id, unit_type, name, "
                    " created_at, updated_at) "
                    "VALUES (:id, :org, :parent, 'faculty', :name, NOW(), NOW())"
                ),
                {
                    "id": str(root_id),
                    "org": str(org_id),
                    "parent": None,
                    "name": "root",
                },
            )
            await session.flush()

            ancestors = await get_org_unit_ancestors(session, root_id)
        finally:
            await trans.rollback()

    assert all(isinstance(a, OrgUnitDTO) for a in ancestors)
    assert [a.id for a in ancestors] == [root_id]
    assert [a.depth for a in ancestors] == [0]
    assert all(a.organization_id == org_id for a in ancestors)
    assert ancestors[0].parent_unit_id is None


@pytest.mark.asyncio
async def test_org_ancestors_missing_unit_returns_empty(test_engine: AsyncEngine) -> None:
    async with AsyncSession(test_engine) as session:
        ancestors = await get_org_unit_ancestors(session, uuid4())
    assert ancestors == []
