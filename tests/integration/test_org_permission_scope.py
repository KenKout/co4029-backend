"""Permissions must resolve against the organization that owns the resource.

The gap this pins down is one level deeper than the membership check added in
"Enforce organization scope on every org-owned route". That check asks *"are
you a member of the org that owns this?"*. The question that actually
authorises the request is *"was your permission granted FOR this org?"*, and
the two come apart the moment somebody belongs to two organizations:

    Alice is a STUDENT in org A            -> member of org A
    Alice is a MANAGER in org B            -> course.update, scope_kind=organization, org B

    PATCH /management/career-paths/{a path in org A}
      require_any_permission("course.update")  passes: she holds the code (in B)
      membership check on org A               passes: she is a member (as a student)
      -> she edits org A's career path, never having been granted authoring there

``load_user_permissions`` cannot see the difference because it flattens role
assignments without reading ``scope_kind`` at all. ``load_org_permissions``
does read it, and these tests are the contract for that.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.features.access_control.queries import (
    load_org_permissions,
    load_user_permissions,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """A session inside a transaction that is always rolled back.

    Every test here inserts organizations, users and role assignments. Rolling
    back rather than deleting keeps them from leaking into the shared database
    even when an assertion fails part-way through.
    """
    engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
    await engine.dispose()


async def _seed(session, *, org_a: uuid.UUID, org_b: uuid.UUID, user: uuid.UUID) -> None:
    """Alice: student in org A, manager (course.update) in org B."""
    await session.execute(
        text(
            "INSERT INTO organizations (id, slug, name, status) "
            "VALUES (:a, :sa, 'Org A', 'active'), (:b, :sb, 'Org B', 'active')"
        ),
        {"a": org_a, "b": org_b, "sa": f"org-a-{org_a.hex[:8]}", "sb": f"org-b-{org_b.hex[:8]}"},
    )
    await session.execute(
        text(
            "INSERT INTO users (id, primary_email, status) "
            "VALUES (:u, :email, 'active')"
        ),
        {"u": user, "email": f"alice-{user.hex[:8]}@example.test"},
    )
    await session.execute(
        text(
            "INSERT INTO organization_memberships (id, user_id, organization_id, status) "
            "VALUES (gen_random_uuid(), :u, :a, 'active')"
        ),
        {"u": user, "a": org_a},
    )

    role_id = (
        await session.execute(
            text("SELECT id FROM roles WHERE code = 'manager' AND deleted_at IS NULL LIMIT 1")
        )
    ).scalar_one()
    await session.execute(
        text(
            "INSERT INTO user_role_assignments "
            "(id, user_id, role_id, scope_kind, organization_id, active_from) "
            "VALUES (gen_random_uuid(), :u, :r, 'organization', :b, NOW() - INTERVAL '1 day')"
        ),
        {"u": user, "r": role_id, "b": org_b},
    )
    await session.flush()


async def test_flat_permission_set_cannot_tell_the_two_orgs_apart(db_session) -> None:
    """Documents the behaviour that makes the org-scoped resolver necessary.

    This is not a bug report against ``load_user_permissions`` — the flat set is
    a legitimate "does this principal hold X anywhere" question, used to gate
    routes before the resource is known. It is only wrong as a *final* answer
    for an org-owned resource.
    """
    org_a, org_b, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _seed(db_session, org_a=org_a, org_b=org_b, user=user)

    flat = await load_user_permissions(db_session, user)
    assert "course.update" in flat, "manager role should carry course.update"


async def test_org_scoped_resolver_grants_only_in_the_granting_org(db_session) -> None:
    org_a, org_b, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _seed(db_session, org_a=org_a, org_b=org_b, user=user)

    in_b = await load_org_permissions(db_session, user, org_b)
    assert "course.update" in in_b, "the grant was made for org B"

    in_a = await load_org_permissions(db_session, user, org_a)
    assert "course.update" not in in_a, (
        "membership in org A must not confer a permission granted in org B"
    )


async def test_global_scope_still_applies_everywhere(db_session) -> None:
    """A globally-granted role is unaffected — that is what 'global' means."""
    org_a, org_b, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _seed(db_session, org_a=org_a, org_b=org_b, user=user)

    role_id = (
        await db_session.execute(
            text("SELECT id FROM roles WHERE code = 'admin' AND deleted_at IS NULL LIMIT 1")
        )
    ).scalar_one()
    await db_session.execute(
        text(
            "INSERT INTO user_role_assignments "
            "(id, user_id, role_id, scope_kind, active_from) "
            "VALUES (gen_random_uuid(), :u, :r, 'global', NOW() - INTERVAL '1 day')"
        ),
        {"u": user, "r": role_id},
    )
    await db_session.flush()

    assert "system.administer" in await load_org_permissions(db_session, user, org_a)
    assert "system.administer" in await load_org_permissions(db_session, user, org_b)


async def test_expired_assignment_is_not_counted(db_session) -> None:
    org_a, org_b, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await _seed(db_session, org_a=org_a, org_b=org_b, user=user)

    await db_session.execute(
        text(
            "UPDATE user_role_assignments SET active_until = NOW() - INTERVAL '1 hour' "
            "WHERE user_id = :u"
        ),
        {"u": user},
    )
    await db_session.flush()

    assert "course.update" not in await load_org_permissions(db_session, user, org_b)
