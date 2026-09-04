"""Integration tests for FR-6.7 audit-trail immutability (migration 0105).

The read side of FR-6.7 was already covered (``/admin/audit/*`` in
``test_admin.py``, middleware persistence in ``test_audit_log.py``). What was
never enforced was the "immutable" half of the requirement, so these tests
drive the database directly rather than the API: the guarantee has to hold for
ANY connection, including one that never goes through FastAPI.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from conftest import SeededUsers
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from abridgeai.core.audit import audit_maintenance


@pytest_asyncio.fixture
async def audit_row(test_engine: AsyncEngine) -> AsyncIterator[uuid.UUID]:
    """One disposable ``http_audit_log`` row, removed via the retention scope."""
    row_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO http_audit_log "
                "(id, request_id, method, path, status_code, latency_ms) "
                "VALUES (:id, :rid, 'GET', '/api/v1/immutability-probe', 200, 5)"
            ),
            {"id": row_id, "rid": uuid.uuid4()},
        )
    yield row_id
    async with test_engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(text("DELETE FROM http_audit_log WHERE id = :id"), {"id": row_id})


# ---------------------------------------------------------------------------
# Append-only event stores
# ---------------------------------------------------------------------------


async def test_http_audit_log_update_is_rejected(
    test_engine: AsyncEngine, audit_row: uuid.UUID
) -> None:
    """Rewriting a logged request must fail -- this is the core FR-6.7 claim."""
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("UPDATE http_audit_log SET status_code = 200 WHERE id = :id"),
                {"id": audit_row},
            )
    assert "append-only" in str(exc.value)


async def test_http_audit_log_update_is_rejected_even_under_maintenance(
    test_engine: AsyncEngine, audit_row: uuid.UUID
) -> None:
    """The retention hatch grants DELETE only.

    If ``audit_maintenance`` also unlocked UPDATE, the whole guarantee would
    reduce to "tamper-proof unless you know the magic GUC" -- retention needs
    to remove rows, never to edit them.
    """

    async def _update_inside_maintenance_scope() -> None:
        # SET LOCAL only holds for its own transaction, so the grant and the
        # UPDATE it is meant to (not) authorise must share one.
        async with test_engine.begin() as conn:
            await audit_maintenance(conn)
            await conn.execute(
                text("UPDATE http_audit_log SET path = '/rewritten' WHERE id = :id"),
                {"id": audit_row},
            )

    with pytest.raises(DBAPIError) as exc:
        await _update_inside_maintenance_scope()
    assert "append-only" in str(exc.value)


async def test_http_audit_log_delete_is_rejected_without_maintenance(
    test_engine: AsyncEngine, audit_row: uuid.UUID
) -> None:
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(text("DELETE FROM http_audit_log WHERE id = :id"), {"id": audit_row})
    assert "app.audit_maintenance" in str(exc.value)

    # Still there: the failed DELETE removed nothing.
    async with test_engine.connect() as conn:
        found = (
            await conn.execute(
                text("SELECT 1 FROM http_audit_log WHERE id = :id"), {"id": audit_row}
            )
        ).first()
    assert found is not None


async def test_http_audit_log_delete_succeeds_under_maintenance(
    test_engine: AsyncEngine,
) -> None:
    """Retention still works -- an unprunable audit store is its own outage."""
    row_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO http_audit_log "
                "(id, request_id, method, path, status_code, latency_ms) "
                "VALUES (:id, :rid, 'GET', '/api/v1/retention-probe', 200, 5)"
            ),
            {"id": row_id, "rid": uuid.uuid4()},
        )

    async with test_engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(text("DELETE FROM http_audit_log WHERE id = :id"), {"id": row_id})

    async with test_engine.connect() as conn:
        found = (
            await conn.execute(text("SELECT 1 FROM http_audit_log WHERE id = :id"), {"id": row_id})
        ).first()
    assert found is None


async def test_maintenance_grant_does_not_outlive_its_transaction(
    test_engine: AsyncEngine, audit_row: uuid.UUID
) -> None:
    """``SET LOCAL`` semantics: the grant cannot leak onto a pooled connection.

    Without this, one retention job would leave every later request on that
    same connection able to delete audit rows.
    """
    async with test_engine.begin() as conn:
        await audit_maintenance(conn)  # granted, then the transaction ends

    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(text("DELETE FROM http_audit_log WHERE id = :id"), {"id": audit_row})
    assert "app.audit_maintenance" in str(exc.value)


async def test_system_setting_changes_update_is_rejected(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The settings change trail is append-only too (ADM-031/033).

    ``reason`` is the column an operator would most want to revise after the
    fact, which is exactly why it must be frozen.
    """
    row_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO system_setting_changes "
                "(id, setting_key, scope, action, after_value_json, reason, actor_id) "
                "VALUES (:id, 'immutability.probe', 'global', 'set', "
                "CAST('1' AS jsonb), 'initial reason', :actor)"
            ),
            {"id": row_id, "actor": seeded_users.admin_id},
        )
    try:
        with pytest.raises(DBAPIError) as exc:
            async with test_engine.begin() as conn:
                await conn.execute(
                    text("UPDATE system_setting_changes SET reason = 'revised' WHERE id = :id"),
                    {"id": row_id},
                )
        assert "append-only" in str(exc.value)
    finally:
        async with test_engine.begin() as conn:
            await audit_maintenance(conn)
            await conn.execute(
                text("DELETE FROM system_setting_changes WHERE id = :id"),
                {"id": row_id},
            )


# ---------------------------------------------------------------------------
# Provenance columns on live entity tables
# ---------------------------------------------------------------------------


async def test_course_created_at_is_immutable(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("UPDATE courses SET created_at = :ts WHERE id = :id"),
                {"ts": datetime.now(tz=UTC) - timedelta(days=365), "id": seeded_users.course_id},
            )
    assert "created_at is immutable" in str(exc.value)


async def test_course_created_by_is_immutable(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Reassigning authorship is the tamper this guard exists to stop."""
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("UPDATE courses SET created_by = :actor WHERE id = :id"),
                {"actor": seeded_users.student_id, "id": seeded_users.course_id},
            )
    assert "created_by is immutable" in str(exc.value)


async def test_user_created_at_is_immutable(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """``users`` is TimestampMixin-only -- no ``created_by`` column at all.

    Covers the key-presence branch in ``audit_columns_immutable()``: a
    plain ``NEW.created_by`` reference would raise on this table for every
    UPDATE, including legitimate ones.
    """
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("UPDATE users SET created_at = :ts WHERE id = :id"),
                {"ts": datetime.now(tz=UTC), "id": seeded_users.student_id},
            )
    assert "created_at is immutable" in str(exc.value)


async def test_ordinary_user_update_still_succeeds(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The provenance guard must not turn live tables read-only.

    ``users`` carries the trigger AND no ``created_by``, so it is the case
    most likely to have been broken by a naive implementation.
    """
    async with test_engine.begin() as conn:
        before = (
            await conn.execute(
                text("SELECT last_login_at FROM users WHERE id = :id"),
                {"id": seeded_users.student_id},
            )
        ).scalar_one()

        stamp = datetime.now(tz=UTC)
        await conn.execute(
            text("UPDATE users SET last_login_at = :ts WHERE id = :id"),
            {"ts": stamp, "id": seeded_users.student_id},
        )
        after = (
            await conn.execute(
                text("SELECT last_login_at FROM users WHERE id = :id"),
                {"id": seeded_users.student_id},
            )
        ).scalar_one()

        assert after != before or before is None
        assert after is not None

        # Restore so the shared session-scoped fixture is left as found.
        await conn.execute(
            text("UPDATE users SET last_login_at = :ts WHERE id = :id"),
            {"ts": before, "id": seeded_users.student_id},
        )


async def test_role_assignment_stays_revocable(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Freezing provenance must not freeze the row.

    Role revocation is a soft-delete UPDATE on ``user_role_assignments`` --
    the very table FR-6.7 names first. If the guard blocked it, the feature
    it protects would stop working.
    """
    assignment_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        role_id = (
            await conn.execute(text("SELECT id FROM roles WHERE code = 'student' LIMIT 1"))
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, granted_by) "
                "VALUES (:id, :uid, :rid, 'organization', :org, :actor)"
            ),
            {
                "id": assignment_id,
                "uid": seeded_users.student_id,
                "rid": role_id,
                "org": seeded_users.organization_id,
                "actor": seeded_users.admin_id,
            },
        )
    try:
        async with test_engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE user_role_assignments "
                    "SET deleted_at = NOW(), deleted_by = :actor WHERE id = :id"
                ),
                {"actor": seeded_users.admin_id, "id": assignment_id},
            )
            revoked = (
                await conn.execute(
                    text("SELECT deleted_at FROM user_role_assignments WHERE id = :id"),
                    {"id": assignment_id},
                )
            ).scalar_one()
        assert revoked is not None

        with pytest.raises(DBAPIError) as exc:
            async with test_engine.begin() as conn:
                await conn.execute(
                    text("UPDATE user_role_assignments SET created_by = :actor WHERE id = :id"),
                    {"actor": seeded_users.student_id, "id": assignment_id},
                )
        assert "created_by is immutable" in str(exc.value)
    finally:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM user_role_assignments WHERE id = :id"),
                {"id": assignment_id},
            )
