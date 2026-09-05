"""Recovery bookkeeping and evaluation results share one JSONB column safely.

``internal_summary_json`` has two independent writers:

* the recovery sweep, which owns ``evaluation_recovery`` (attempt counter +
  dispatch timestamp);
* the evaluator, which owns everything else (``total_score``,
  ``rubric_aggregated``, ``evaluated_at``, and the ``evaluation_failure`` note it
  clears on success).

They overlap in time by design. The sweep's grace window is 15 minutes and
``job_timeout`` is 20, so an evaluator can publish while the sweep is between its
candidate query and its enqueue. The sweep used to read the whole column into a
Python dict, mutate it, and assign it back — twice, once to charge the attempt and
once to refund it on a dispatch failure. Both writes blind-overwrote the column
from a snapshot taken before the evaluator committed, so a verdict published in
that window lost its rubric totals and got its stale failure note resurrected.

These tests drive real transactions against Postgres, because the fix IS the SQL:
``jsonb_set`` on one key, guarded by ``pass_verdict IS NULL`` (charge) and by the
attempt count we wrote (refund). A test that used the ORM would prove nothing.
"""

from __future__ import annotations

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
async def recovery_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """One terminal, ungraded session with an empty summary."""
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
            {"id": org_id, "slug": f"rm-{suffix}", "name": "Recovery Meta Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"rm-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'RM Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"rm-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'RM Interview', 'published', 'neutral', "
                ":teacher, 'slug-' || uuid_generate_v4()::text)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, ended_at, onboarding_stage, "
                "interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'failed', 'text', now(), now(), now(), "
                "'completed', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "student_id": student_id, "course_id": course_id}

    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"), {"t": teacher_id, "s": student_id}
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _summary(test_engine: AsyncEngine, session_id: UUID) -> dict[str, Any]:
    async with test_engine.begin() as conn:
        return (
            await conn.execute(
                text("SELECT internal_summary_json FROM interview_sessions WHERE id = :s"),
                {"s": session_id},
            )
        ).scalar_one()


async def _set_summary(
    test_engine: AsyncEngine, session_id: UUID, summary: dict[str, Any]
) -> None:
    import json

    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions SET internal_summary_json = CAST(:s AS jsonb) "
                " WHERE id = :id"
            ),
            {"s": json.dumps(summary), "id": session_id},
        )


def _maker(test_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)


async def test_charging_an_attempt_starts_the_counter_at_one(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = recovery_probe["session_id"]
    now = datetime.now(UTC)

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(db, session_id, now=now)

    assert attempt == 1
    recovery = (await _summary(test_engine, session_id))["evaluation_recovery"]
    assert recovery["attempts"] == 1
    assert recovery["last_attempt_at"] == now.isoformat()


async def test_charging_increments_and_returns_the_new_count(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The returned count is what the per-attempt job ID is built from."""
    session_id = recovery_probe["session_id"]
    await _set_summary(
        test_engine,
        session_id,
        {"evaluation_recovery": {"attempts": 2, "last_attempt_at": "2026-01-01T00:00:00+00:00"}},
    )

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )

    assert attempt == 3
    assert (await _summary(test_engine, session_id))["evaluation_recovery"]["attempts"] == 3


async def test_charging_preserves_everything_the_evaluator_owns(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """THE BUG: a whole-dict write from a stale snapshot deleted real results.

    The evaluator publishes between the sweep's candidate query and its charge, so
    the sweep's snapshot of this column predates the verdict.
    """
    session_id = recovery_probe["session_id"]
    await _set_summary(
        test_engine,
        session_id,
        {
            "total_score": 82.5,
            "rubric_aggregated": {"clarity": 4.0},
            "evaluated_at": "2026-09-05T12:00:00+00:00",
        },
    )

    async with _maker(test_engine)() as db:
        await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )

    summary = await _summary(test_engine, session_id)
    assert summary["total_score"] == 82.5, "the attempt counter clobbered the rubric total"
    assert summary["rubric_aggregated"] == {"clarity": 4.0}
    assert summary["evaluated_at"] == "2026-09-05T12:00:00+00:00"
    assert summary["evaluation_recovery"]["attempts"] == 1


async def test_a_graded_session_cannot_be_charged_an_attempt(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """``None`` tells the sweep to skip: no attempt spent, no second grader."""
    session_id = recovery_probe["session_id"]
    async with test_engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_sessions SET pass_verdict = FALSE WHERE id = :s"),
            {"s": session_id},
        )

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )

    assert attempt is None
    assert "evaluation_recovery" not in await _summary(test_engine, session_id)


async def test_a_refund_removes_a_counter_it_created(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = recovery_probe["session_id"]
    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )
        assert attempt == 1
        refunded = await sessions_queries.refund_evaluation_recovery_attempt(
            db, session_id, attempt=1, previous_recovery=None
        )

    assert refunded is True
    assert "evaluation_recovery" not in await _summary(test_engine, session_id)


async def test_a_refund_restores_the_previous_count(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = recovery_probe["session_id"]
    previous = {"attempts": 1, "last_attempt_at": "2026-01-01T00:00:00+00:00"}
    await _set_summary(test_engine, session_id, {"evaluation_recovery": dict(previous)})

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )
        assert attempt == 2
        assert await sessions_queries.refund_evaluation_recovery_attempt(
            db, session_id, attempt=2, previous_recovery=previous
        )

    assert (await _summary(test_engine, session_id))["evaluation_recovery"] == previous


async def test_a_refund_does_not_delete_a_verdict_published_mid_dispatch(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """THE OTHER HALF OF THE BUG (P2 #2).

    Sequence: the sweep charges an attempt, its enqueue hangs on a dying Redis,
    the evaluator finishes and publishes, then the enqueue raises and the sweep
    refunds. Restoring a whole pre-enqueue snapshot of the column deleted the
    metadata the evaluator had just written and resurrected the failure note it
    had cleared.
    """
    session_id = recovery_probe["session_id"]
    await _set_summary(
        test_engine,
        session_id,
        {"evaluation_failure": {"message": "LLM 503"}},
    )

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )
    assert attempt == 1

    # The evaluator wins the race: publishes, and clears its own failure note.
    await _set_summary(
        test_engine,
        session_id,
        {
            "evaluation_recovery": (await _summary(test_engine, session_id))["evaluation_recovery"],
            "total_score": 91.0,
            "evaluated_at": "2026-09-05T12:00:00+00:00",
        },
    )
    async with test_engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_sessions SET pass_verdict = TRUE WHERE id = :s"),
            {"s": session_id},
        )

    # Only now does the dispatch fail and the sweep refund.
    async with _maker(test_engine)() as db:
        await sessions_queries.refund_evaluation_recovery_attempt(
            db, session_id, attempt=1, previous_recovery=None
        )

    summary = await _summary(test_engine, session_id)
    assert summary["total_score"] == 91.0, "the refund deleted the published verdict's metadata"
    assert summary["evaluated_at"] == "2026-09-05T12:00:00+00:00"
    assert "evaluation_failure" not in summary, (
        "the refund resurrected a failure note the evaluator had cleared"
    )
    assert "evaluation_recovery" not in summary


async def test_a_refund_is_skipped_once_a_later_sweep_has_charged_its_own(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Un-charging someone else's attempt would reopen the infinite-retry loop."""
    session_id = recovery_probe["session_id"]
    async with _maker(test_engine)() as db:
        assert await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        ) == 1
        # A later sweep charges its own attempt before our refund lands.
        assert await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC) + timedelta(minutes=5)
        ) == 2
        refunded = await sessions_queries.refund_evaluation_recovery_attempt(
            db, session_id, attempt=1, previous_recovery=None
        )

    assert refunded is False
    assert (await _summary(test_engine, session_id))["evaluation_recovery"]["attempts"] == 2


async def test_a_malformed_attempt_counter_is_read_as_zero(
    recovery_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """One hand-edited row must not raise out of the sweep for everyone else."""
    session_id = recovery_probe["session_id"]
    await _set_summary(
        test_engine, session_id, {"evaluation_recovery": {"attempts": "lots"}}
    )

    async with _maker(test_engine)() as db:
        attempt = await sessions_queries.stamp_evaluation_recovery_attempt(
            db, session_id, now=datetime.now(UTC)
        )

    assert attempt == 1
