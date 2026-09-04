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


# ---------------------------------------------------------------------------
# Assessment audit stores (migration 0106)
# ---------------------------------------------------------------------------
#
# These two were documented as append-only long before anything enforced it:
# ``assessment_integrity_events`` is the proctoring log whose whole evidentiary
# value is that a participant cannot retroactively remove a tab-switch, and
# ``quiz_audit_events`` is described in its router as an append-only trail.
# 0105 guarded http_audit_log + system_setting_changes and left these two on
# convention alone; 0106 closed that. Driven at the DB level for the same reason
# as the tests above: the guarantee must hold for ANY connection.


@pytest_asyncio.fixture
async def probe_quiz(
    test_engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[tuple[uuid.UUID, uuid.UUID]]:
    """A throwaway ``(quiz_id, attempt_id)`` for the assessment audit rows.

    Both stores need real parents: ``quiz_audit_events.quiz_id`` is NOT NULL,
    and ``assessment_integrity_events`` carries a CHECK requiring
    ``quiz_attempt_id IS NOT NULL`` (and ``interview_session_id IS NULL``) when
    ``assessment_kind = 'quiz'`` — the kind and its subject id must agree.

    ``position`` is parked at 9001 rather than 1: the seeded course is shared
    session-wide and another test taking position 1 would collide on
    ``modules_course_id_position_key``.
    """
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    attempt_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:mid, :cid, 'Audit probe module', 9001, 'draft')"
            ),
            {"mid": module_id, "cid": seeded_users.course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, slug, status) "
                "VALUES (:qid, :cid, :mid, 'Audit probe quiz', :slug, 'draft')"
            ),
            {
                "qid": quiz_id,
                "cid": seeded_users.course_id,
                "mid": module_id,
                "slug": f"audit-probe-{quiz_id.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts (id, quiz_id, student_id, attempt_number, status) "
                "VALUES (:aid, :qid, :sid, 1, 'submitted')"
            ),
            {"aid": attempt_id, "qid": quiz_id, "sid": seeded_users.student_id},
        )
    yield quiz_id, attempt_id
    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM quiz_attempts WHERE id = :id"), {"id": attempt_id})
        await conn.execute(text("DELETE FROM quizzes WHERE id = :id"), {"id": quiz_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})


@pytest_asyncio.fixture
async def integrity_event_row(
    test_engine: AsyncEngine,
    seeded_users: SeededUsers,
    probe_quiz: tuple[uuid.UUID, uuid.UUID],
) -> AsyncIterator[uuid.UUID]:
    """One disposable ``assessment_integrity_events`` row."""
    _, attempt_id = probe_quiz
    row_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO assessment_integrity_events "
                "(id, assessment_kind, quiz_attempt_id, student_id, event_type, "
                " severity, metadata_json) "
                "VALUES (:id, 'quiz', :aid, :sid, 'tab_switch', 'warning', '{}'::jsonb)"
            ),
            {"id": row_id, "aid": attempt_id, "sid": seeded_users.student_id},
        )
    yield row_id
    async with test_engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(
            text("DELETE FROM assessment_integrity_events WHERE id = :id"), {"id": row_id}
        )


async def test_integrity_event_update_is_rejected(
    test_engine: AsyncEngine, integrity_event_row: uuid.UUID
) -> None:
    """Downgrading a recorded proctoring signal must be impossible.

    ``severity`` is the field an interested party would most want to soften
    after the fact, which is exactly why UPDATE has no bypass at all.
    """
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE assessment_integrity_events SET severity = 'info' WHERE id = :id"
                ),
                {"id": integrity_event_row},
            )
    assert "append-only" in str(exc.value)
    assert "UPDATE is never permitted" in str(exc.value)


async def test_integrity_event_delete_needs_the_maintenance_scope(
    test_engine: AsyncEngine, integrity_event_row: uuid.UUID
) -> None:
    """A plain DELETE is refused; the retention scope is the only way through."""
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM assessment_integrity_events WHERE id = :id"),
                {"id": integrity_event_row},
            )
    assert "app.audit_maintenance" in str(exc.value)


@pytest_asyncio.fixture
async def quiz_audit_event_row(
    test_engine: AsyncEngine,
    seeded_users: SeededUsers,
    probe_quiz: tuple[uuid.UUID, uuid.UUID],
) -> AsyncIterator[uuid.UUID]:
    """One disposable ``quiz_audit_events`` row on the probe quiz."""
    quiz_id, _ = probe_quiz
    row_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_audit_events "
                "(id, event_name, quiz_id, actor_user_id, payload_json) "
                "VALUES (:id, 'attempt_started', :qid, :actor, '{}'::jsonb)"
            ),
            {"id": row_id, "qid": quiz_id, "actor": seeded_users.student_id},
        )
    yield row_id
    async with test_engine.begin() as conn:
        await audit_maintenance(conn)
        await conn.execute(text("DELETE FROM quiz_audit_events WHERE id = :id"), {"id": row_id})


async def test_quiz_audit_event_update_is_rejected(
    test_engine: AsyncEngine, quiz_audit_event_row: uuid.UUID
) -> None:
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("UPDATE quiz_audit_events SET event_name = 'rewritten' WHERE id = :id"),
                {"id": quiz_audit_event_row},
            )
    assert "append-only" in str(exc.value)


async def test_quiz_audit_event_delete_needs_the_maintenance_scope(
    test_engine: AsyncEngine, quiz_audit_event_row: uuid.UUID
) -> None:
    with pytest.raises(DBAPIError) as exc:
        async with test_engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM quiz_audit_events WHERE id = :id"),
                {"id": quiz_audit_event_row},
            )
    assert "app.audit_maintenance" in str(exc.value)


async def test_every_guarded_table_has_a_retention_entry(
    test_engine: AsyncEngine,
) -> None:
    """The retention sweep must cover every table the trigger guards.

    A guarded table with no retention entry is the failure mode this asserts
    against: it can never be pruned by any code path, so it grows forever with
    no way to stop it short of a manual migration. Discovered from the live
    catalog rather than a hard-coded list, so a table guarded by a FUTURE
    migration fails this test until it is given a window.

    ``system_setting_changes`` is the deliberate exception: it is the
    configuration-change ledger, bounded by how often an admin retunes a
    setting, and losing it would erase the provenance the settings console
    reports. It is guarded but intentionally never pruned.
    """
    from abridgeai.core.audit.retention import RETAINED_TABLES

    async with test_engine.begin() as conn:
        guarded = {
            row
            for row in (
                await conn.execute(
                    text(
                        "SELECT c.relname FROM pg_trigger tg "
                        "JOIN pg_proc p ON p.oid = tg.tgfoid "
                        "JOIN pg_class c ON c.oid = tg.tgrelid "
                        "WHERE NOT tg.tgisinternal AND p.proname = 'audit_log_immutable'"
                    )
                )
            ).scalars()
        }

    pruned = {spec.table for spec in RETAINED_TABLES}
    never_pruned_by_design = {"system_setting_changes"}
    assert guarded == pruned | never_pruned_by_design, (
        f"guarded={sorted(guarded)} pruned={sorted(pruned)}"
    )
