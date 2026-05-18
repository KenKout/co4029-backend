from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.access_control.api.public import (
    OrgUnitDTO,
    can_manage_course,
    get_org_unit_ancestors,
    require_any_permission,
    require_course_permission,
    require_permission,
)
from abridgeai.features.access_control import policies as _policies


def test_decorator_reexport_is_true_alias() -> None:
    assert require_permission is _policies.require_permission
    assert require_any_permission is _policies.require_any_permission
    assert require_course_permission is _policies.require_course_permission
    assert can_manage_course is _policies.can_manage_course


def test_decorator_reexport_usable_from_sibling() -> None:
    dep = require_permission("some.code")
    assert callable(dep)


@pytest.mark.asyncio
async def test_org_ancestors_4_level_tree(test_engine: AsyncEngine) -> None:
    org_id = uuid4()
    root_id = uuid4()
    l1_id = uuid4()
    l2_id = uuid4()
    leaf_id = uuid4()

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
            for unit_id, parent_id, name in (
                (root_id, None, "root"),
                (l1_id, root_id, "l1"),
                (l2_id, l1_id, "l2"),
                (leaf_id, l2_id, "leaf"),
            ):
                await session.execute(
                    text(
                        "INSERT INTO org_units "
                        "(id, organization_id, parent_unit_id, unit_type, name, "
                        " created_at, updated_at) "
                        "VALUES (:id, :org, :parent, 'department', :name, NOW(), NOW())"
                    ),
                    {
                        "id": str(unit_id),
                        "org": str(org_id),
                        "parent": str(parent_id) if parent_id else None,
                        "name": name,
                    },
                )
            await session.flush()

            ancestors = await get_org_unit_ancestors(session, leaf_id)
        finally:
            await trans.rollback()

    assert all(isinstance(a, OrgUnitDTO) for a in ancestors)
    assert [a.id for a in ancestors] == [leaf_id, l2_id, l1_id, root_id]
    assert [a.depth for a in ancestors] == [0, 1, 2, 3]
    assert all(a.organization_id == org_id for a in ancestors)
    assert ancestors[0].parent_unit_id == l2_id
    assert ancestors[-1].parent_unit_id is None


@pytest.mark.asyncio
async def test_org_ancestors_missing_unit_returns_empty(test_engine: AsyncEngine) -> None:
    async with AsyncSession(test_engine) as session:
        ancestors = await get_org_unit_ancestors(session, uuid4())
    assert ancestors == []
