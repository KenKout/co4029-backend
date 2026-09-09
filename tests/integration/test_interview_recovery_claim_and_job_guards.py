"""The recovery stamp must re-check, atomically, what the candidate query only
approximated.

``list_pending_evaluation_sessions`` runs minutes before the stamp: its idea of
the session (verdict null, under the ceiling, no claim) is a SNAPSHOT. Between
that query and the charge an evaluator can publish, a claim can be taken, and a
previous recovery job can start running. The stamp is the last gate before an
attempt is charged and a job is dispatched, so IT must hold the guards:

* a verdict published in the window refuses the charge (already pinned in
  ``test_interview_recovery_metadata_isolation.py``);
* a LIVE claim refuses the charge — a grader owns the session and a second one
  would race it;
* a durable ACTIVE recovery job (dispatching/queued/running/retrying, from the
  ``evaluation_recovery.current`` bookkeeping) refuses the charge — the last
  re-drive is still working, and the attempt is not the sweep's to spend.

These tests drive real transactions against Postgres: the guards live in the
UPDATE's WHERE clause, and only a concurrent-writer test can prove they hold.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from abridgeai.features.interviews.queries import sessions as sessions_queries

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def recovery_guard_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """Minimal FK chain plus one terminal, ungraded session with no claim."""
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
            {"id": org_id, "slug": f"rg-{suffix}", "name": "Recovery Guard Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"rg-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'RG Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"rg-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'RG Interview', 'published', 'neutral', "
                ":teacher, 'slug-' || uuid_generate_v4()::text)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, ended_at, onboarding_stage, "
                "interview_language, internal_summary_json) "
                "VALUES (:id, :cfg, :student, 1, 'failed', 'text', now(), now(), now(), "
                "'completed', 'en', '{}'::jsonb)"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "student_id": student_id}

    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"), {"t": teacher_id, "s": student_id}
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


def _maker(test_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)


async def _set_claim(
    test_engine: AsyncEngine,
    session_id: UUID,
    *,
    token: uuid.UUID | None,
    expires_at: datetime | None,
) -> None:
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions SET evaluation_claim_token = :token, "
                "evaluation_claim_expires_at = :expires_at WHERE id = :id"
            ),
            {
                "token": token,
                "expires_at": expires_at,
                "id": session_id,
            },
        )


async def _set_recovery(
    test_engine: AsyncEngine, session_id: UUID, recovery: dict[str, Any]
) -> None:
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "SET internal_summary_json = jsonb_set("
                "  COALESCE(internal_summary_json, '{}'::jsonb), "
                "  '{evaluation_recovery}', CAST(:r AS jsonb), true) "
                "WHERE id = :id"
            ),
            {"r": json.dumps(recovery), "id": session_id},
        )


async def _summary(test_engine: AsyncEngine, session_id: UUID) -> dict[str, Any]:
    async with test_engine.begin() as conn:
        return (
            await conn.execute(
                text("SELECT internal_summary_json FROM interview_sessions WHERE id = :s"),
                {"s": session_id},
            )
        ).scalar_one()


async def _stamp(test_engine: AsyncEngine, session_id: UUID) -> int | None:
    now = datetime.now(UTC)
    async with _maker(test_engine)() as db:
        return await sessions_queries.stamp_evaluation_recovery_attempt(db, session_id, now=now)


async def test_a_live_claim_is_never_charged(
    recovery_guard_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = recovery_guard_probe["session_id"]
    await _set_claim(
        test_engine,
        session_id,
        token=uuid.uuid4(),
        expires_at=datetime.now(UTC) + timedelta(minutes=20),
    )

    attempt = await _stamp(test_engine, session_id)

    assert attempt is None, "the sweep charged a second grader against a live claim"
    summary = await _summary(test_engine, session_id)
    assert "evaluation_recovery" not in summary


async def test_an_expired_claim_does_not_block_the_charge(
    recovery_guard_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = recovery_guard_probe["session_id"]
    await _set_claim(
        test_engine,
        session_id,
        token=uuid.uuid4(),
        expires_at=datetime.now(UTC) - timedelta(minutes=5),
    )

    attempt = await _stamp(test_engine, session_id)

    assert attempt == 1, "a lapsed lease means the owner is dead; the sweep must proceed"


async def test_a_null_claim_never_blocks_the_charge(
    recovery_guard_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = recovery_guard_probe["session_id"]
    await _set_claim(test_engine, session_id, token=None, expires_at=None)

    assert await _stamp(test_engine, session_id) == 1


@pytest.mark.parametrize(
    ("phase", "chargeable"),
    [
        ("dispatching", False),
        ("queued", False),
        ("running", False),
        ("retrying", False),
        ("succeeded", True),
        ("failed", True),
        ("missing", True),
    ],
)
async def test_a_durable_active_job_refuses_the_charge(
    recovery_guard_probe: dict[str, Any],
    test_engine: AsyncEngine,
    phase: str,
    chargeable: bool,
) -> None:
    """The previous recovery's job is still working (or freshly terminal).

    An ACTIVE phase means the last re-drive never settled: charging now would
    send a second grader against the same transcript. Only a TERMINAL phase
    (or no ``current`` at all) leaves the budget free.
    """
    session_id = recovery_guard_probe["session_id"]
    await _set_recovery(
        test_engine,
        session_id,
        {
            "attempts": 1,
            "last_attempt_at": "2026-01-01T00:00:00+00:00",
            "current": {
                "attempt": 1,
                "job_id": f"interview-evaluation:{session_id}:recover-1",
                "phase": phase,
                "dispatched_at": "2026-01-01T00:00:00+00:00",
            },
        },
    )

    attempt = await _stamp(test_engine, session_id)

    if chargeable:
        assert attempt == 2, f"phase={phase} is terminal; the sweep may charge"
    else:
        assert attempt is None, f"phase={phase} is active; the sweep must not charge"
        summary = await _summary(test_engine, session_id)
        assert summary["evaluation_recovery"]["attempts"] == 1


async def test_the_stamp_writes_the_current_dispatching_record(
    recovery_guard_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Charging an attempt reserves it: the row records the dispatch in flight."""
    session_id = recovery_guard_probe["session_id"]
    now = datetime.now(UTC)

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(db, session_id, now=now)

    assert attempt == 1
    recovery = (await _summary(test_engine, session_id))["evaluation_recovery"]
    assert recovery["attempts"] == 1
    current = recovery.get("current") or {}
    assert current.get("attempt") == 1
    assert current.get("phase") == "dispatching"
    assert current.get("job_id") == f"interview-evaluation:{session_id}:recover-1"
