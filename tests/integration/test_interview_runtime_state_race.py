"""The runtime-state optimistic lock must survive a real concurrent turn.

A retried REST call and a LiveKit callback can process the same interview turn
concurrently. The lock exists so only one of them advances the state; if both
land, one turn is silently discarded — its hint counters, security attempt counts
and phase progress overwritten by a turn that never saw them.

The dangerous interleaving is specific, and a naive test does NOT reach it:

1. T1 reads version N and issues its UPDATE, holding the row lock uncommitted.
2. T2's version check runs *inside that window*. Under READ COMMITTED it still
   SELECTs the OLD row, so the check passes.
3. T2's UPDATE blocks on T1's lock, then proceeds once T1 commits.

With the check in a preceding SELECT, step 3 wrote anyway. With the check in the
UPDATE's own WHERE clause, Postgres re-evaluates it against the committed row
after the lock is released, matches zero rows, and the race is caught.

These tests drive real concurrent transactions, so they build their own sessions
from the engine rather than sharing the per-test session fixture.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """A minimal FK chain plus one ``in_progress`` session and no runtime state.

    Built rather than discovered: the test database is empty, and a test that
    silently skips because it found no suitable row is a test that proves
    nothing.
    """
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    session_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"rs-{suffix}", "name": "Race Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"rs-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'RS Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"rs-course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'published')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, persona, created_by, slug) "
                "VALUES (:id, :course, :module, 'RS Interview', 'published', 'neutral', "
                ":teacher, 'slug-' || uuid_generate_v4()::text)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'text', now(), "
                "'completed', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "config_id": config_id}

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_runtime_states WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"), {"t": teacher_id, "s": student_id}
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _read_row(test_engine: AsyncEngine, session_id: UUID) -> Any:
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as db:
        return (
            await db.execute(
                text(
                    """
                    SELECT state_version,
                           last_turn_idempotency_key AS turn_key,
                           state_json->>'hint_level' AS hint_level
                      FROM interview_runtime_states
                     WHERE session_id = :s
                    """
                ),
                {"s": session_id},
            )
        ).first()


async def test_a_turn_cannot_be_lost_to_a_concurrent_save(
    probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The regression: T2's guard passes inside T1's lock window, then blocks."""
    session_id = probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as seed:
        await state_repo.load_or_init(seed, session_id)
        await seed.commit()

    winner = InterviewRuntimeStateData()
    winner.hint_level = 5
    loser = InterviewRuntimeStateData()
    loser.hint_level = 0

    t1 = maker()
    t2 = maker()
    try:
        loaded = await state_repo.load_or_init(t1, session_id)
        await state_repo.save(
            t1,
            session_id,
            winner,
            expected_version=loaded.version,
            turn_idempotency_key="turn-A",
        )
        # T1 now holds the row lock with an uncommitted UPDATE.

        async def competing_turn() -> str:
            loaded_two = await state_repo.load_or_init(t2, session_id)
            # This guard read happens inside T1's window and sees the old row.
            await state_repo.save(
                t2,
                session_id,
                loser,
                expected_version=loaded_two.version,
                turn_idempotency_key="turn-B",
            )
            await t2.commit()
            return "committed"

        task = asyncio.create_task(competing_turn())
        await asyncio.sleep(0.4)  # let T2 reach its blocking UPDATE
        assert not task.done(), "T2 should be blocked on T1's row lock"

        await t1.commit()

        with pytest.raises(state_repo.StaleStateError):
            await asyncio.wait_for(task, timeout=15)
    finally:
        await t1.close()
        await t2.close()

    row = await _read_row(test_engine, session_id)
    assert row is not None
    assert row.state_version == 1, "exactly one turn should have advanced the state"
    assert row.turn_key == "turn-A", (
        "the losing turn overwrote the winner's idempotency key — a replay of "
        "turn-A would no longer be recognised as a duplicate"
    )
    assert row.hint_level == "5", "the winning turn's state was overwritten"


async def test_a_sequential_stale_save_is_still_rejected(
    probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The simple case must keep working: read v0, someone else commits, save."""
    session_id = probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as seed:
        await state_repo.load_or_init(seed, session_id)
        await seed.commit()

    async with maker() as t1, maker() as t2:
        stale = await state_repo.load_or_init(t2, session_id)  # reads v0

        fresh = await state_repo.load_or_init(t1, session_id)
        await state_repo.save(
            t1,
            session_id,
            InterviewRuntimeStateData(),
            expected_version=fresh.version,
            turn_idempotency_key="turn-A",
        )
        await t1.commit()

        with pytest.raises(state_repo.StaleStateError) as caught:
            await state_repo.save(
                t2,
                session_id,
                InterviewRuntimeStateData(),
                expected_version=stale.version,
                turn_idempotency_key="turn-B",
            )
        assert caught.value.expected == 0
        assert caught.value.actual == 1


async def test_a_successful_save_advances_and_records_the_turn(
    probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Guard-in-the-UPDATE must not break the ordinary happy path.

    Also pins that the write is visible to a re-read in the SAME transaction —
    a bulk UPDATE bypasses the ORM identity map, so a cached instance would
    otherwise still serve pre-save values and the next turn would rewind.
    """
    session_id = probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as own:
        loaded = await state_repo.load_or_init(own, session_id)
        data = InterviewRuntimeStateData()
        data.hint_level = 2
        new_version = await state_repo.save(
            own,
            session_id,
            data,
            expected_version=loaded.version,
            turn_idempotency_key="turn-1",
        )
        assert new_version == loaded.version + 1

        again = await state_repo.load_or_init(own, session_id)
        assert again.version == new_version, "the save must be visible to a re-read"
        assert again.last_turn_idempotency_key == "turn-1"
        assert again.data.hint_level == 2
        assert state_repo.is_duplicate_turn(again, "turn-1") is True
        assert state_repo.is_duplicate_turn(again, "turn-2") is False
        await own.commit()


async def test_saving_without_a_row_reports_a_lost_race(
    probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """No row at all is reported as ``actual == -1``, not a crash."""
    session_id = probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as own:
        with pytest.raises(state_repo.StaleStateError) as caught:
            await state_repo.save(
                own,
                session_id,
                InterviewRuntimeStateData(),
                expected_version=0,
                turn_idempotency_key="turn-1",
            )
        assert caught.value.actual == -1
