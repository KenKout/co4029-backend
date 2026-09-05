"""Two finishes for one session must agree on the terminal status.

Step 4 of the debug plan asked for contract tests on the realtime edges before
concluding anything was broken. This file covers the finish/timeout edge, and it
found a real one.

``submit_session`` guards re-entry with a plain Python read-then-write::

    session = await _require_session(db, session_id)
    if session.status != "in_progress":
        return session
    ...
    session.status = <derived>
    await db.commit()

Under READ COMMITTED, two callers that both read ``in_progress`` both pass that
guard, and the second commit overwrites the first. The two callers exist in
production and can fire within the same second:

* the student pressing finish (``POST /interview-sessions/{id}/finish``);
* the native agent's hard-stop timer calling ``finalize_session``
  (``realtime/orchestration_bridge``), plus the deadline path with
  ``reason='timed_out'``.

That matters because the derived status differs per caller: a natural finish is
``completed``, a deadline finish is ``timed_out``, and an onboarding-only run is
``abandoned``. Whichever transaction commits last wins, so a student who
answered questions and pressed finish can end up recorded as ``timed_out`` (or
the reverse), and ``ended_at`` is likewise decided by scheduling.

Compare ``queries.sessions.finalize_expired_in_progress_session``, which the
deadline SWEEP already uses: the status predicate lives inside the UPDATE's WHERE
clause and it returns the status only when that caller changed the row. The sweep
was hardened; this path was not.

The evaluation enqueue is NOT part of the defect: both callers use the
session-scoped job ID, which ARQ deduplicates.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from abridgeai.core.security import CurrentUser
from abridgeai.features.interviews.services import taking as taking_service

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def finish_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """One live session that has reached the assessment and answered a question."""
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    session_id = uuid.uuid4()
    question_id = uuid.uuid4()
    session_question_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"fr-{suffix}", "name": "Finish Race Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"fr-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'FR Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"fr-course-{suffix}"},
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
                "(id, course_id, module_id, title, status, persona, created_by, slug, "
                "time_limit_minutes) "
                "VALUES (:id, :course, :module, 'FR Interview', 'published', 'neutral', "
                ":teacher, 'slug-' || uuid_generate_v4()::text, 30)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, position, question_type, prompt_text, review_status) "
                "VALUES (:id, :cfg, 1, 'technical', 'Explain indexing.', 'approved')"
            ),
            {"id": question_id, "cfg": config_id},
        )
        # assessment_started_at well in the past so the timed_out branch's
        # deadline assertion is satisfied without sleeping.
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'text', "
                "now() - interval '90 minutes', now() - interval '90 minutes', "
                "'completed', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_session_questions "
                "(id, session_id, interview_question_id, sequence_no) "
                "VALUES (:id, :s, :q, 1)"
            ),
            {"id": session_question_id, "s": session_id, "q": question_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_session_messages "
                "(id, session_id, session_question_id, role, content_text) "
                "VALUES (uuid_generate_v4(), :s, :sq, 'user', 'B-trees keep lookups log n.')"
            ),
            {"s": session_id, "sq": session_question_id},
        )

    yield {"session_id": session_id, "student_id": student_id}

    async with test_engine.begin() as conn:
        for stmt, params in (
            ("DELETE FROM interview_session_messages WHERE session_id = :s", {"s": session_id}),
            ("DELETE FROM interview_session_questions WHERE session_id = :s", {"s": session_id}),
            ("DELETE FROM interview_sessions WHERE id = :s", {"s": session_id}),
            ("DELETE FROM interview_questions WHERE interview_config_id = :c", {"c": config_id}),
            ("DELETE FROM interview_configs WHERE id = :c", {"c": config_id}),
            ("DELETE FROM modules WHERE id = :m", {"m": module_id}),
            ("DELETE FROM courses WHERE id = :c", {"c": course_id}),
            ("DELETE FROM users WHERE id IN (:t, :s)", {"t": teacher_id, "s": student_id}),
            ("DELETE FROM organizations WHERE id = :o", {"o": org_id}),
        ):
            await conn.execute(text(stmt), params)


def _actor(student_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=student_id, session_id=uuid.uuid4(), permissions=frozenset())


async def _status_and_ended_at(test_engine: AsyncEngine, session_id: uuid.UUID) -> Any:
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as db:
        return (
            await db.execute(
                text("SELECT status, ended_at FROM interview_sessions WHERE id = :s"),
                {"s": session_id},
            )
        ).first()


async def test_a_natural_finish_is_not_relabelled_by_a_concurrent_timeout(
    finish_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The student answered and pressed finish; the hard-stop timer fired too.

    The interleaving is forced rather than left to ``asyncio.gather``: both
    transactions must load the session while it is still ``in_progress``, THEN
    write. Gather alone lets one coroutine reach its commit before the other
    reads, so a gathered version of this test passes even against the bug.

    Measured against the unfixed code, T2 relabelled ``completed`` -> ``timed_out``.
    """
    session_id = finish_probe["session_id"]
    actor = _actor(finish_probe["student_id"])
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)

    t1 = maker()
    t2 = maker()
    try:
        # Both transactions observe the live session before either writes.
        first_seen = await taking_service._require_session(t1, session_id)
        second_seen = await taking_service._require_session(t2, session_id)
        assert first_seen.status == "in_progress"
        assert second_seen.status == "in_progress"

        await taking_service.submit_session(t1, session_id, actor, arq_pool=None, reason="natural")
        landed = await _status_and_ended_at(test_engine, session_id)
        assert landed.status == "completed"

        # T2 still holds an instance it loaded as in_progress.
        await taking_service.submit_session(
            t2,
            session_id,
            actor,
            arq_pool=None,
            reason="timed_out",
        )
    finally:
        await t1.close()
        await t2.close()

    final = await _status_and_ended_at(test_engine, session_id)
    assert final.status == "completed", (
        "the student's own finish was relabelled by a stale concurrent submit"
    )
    assert final.ended_at == landed.ended_at, "ended_at must not move after terminalization"


async def test_a_second_submit_never_moves_ended_at(
    finish_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """``ended_at`` anchors the recovery sweep's grace window.

    ``list_pending_evaluation_sessions`` selects on ``ended_at <= cutoff``, so a
    later write pushing it forward would slide the session back out of the
    recovery window.
    """
    session_id = finish_probe["session_id"]
    actor = _actor(finish_probe["student_id"])
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)

    async with maker() as db:
        await taking_service.submit_session(db, session_id, actor, arq_pool=None)
    first = await _status_and_ended_at(test_engine, session_id)

    await asyncio.sleep(0.05)
    async with maker() as db:
        await taking_service.submit_session(db, session_id, actor, arq_pool=None)
    second = await _status_and_ended_at(test_engine, session_id)

    assert second.ended_at == first.ended_at
    assert second.status == first.status


async def test_only_the_caller_that_terminalized_enqueues_evaluation(
    finish_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The enqueue decision belongs to exactly one caller.

    Both callers currently enqueue, and ARQ's session-scoped job ID is what saves
    the system from grading twice. That is a coincidence of the job-ID choice, not
    a property of this function, so pin the stronger contract: only the caller
    that actually changed the row acts on it. The interleaving is forced for the
    same reason as the test above.
    """
    session_id = finish_probe["session_id"]
    actor = _actor(finish_probe["student_id"])
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    enqueued: list[uuid.UUID] = []

    class _RecordingPool:
        async def enqueue_job(
            self, _task: str, _actor_id: Any, sid: uuid.UUID, **_kw: Any
        ) -> object:
            enqueued.append(sid)
            return object()

    t1 = maker()
    t2 = maker()
    try:
        await taking_service._require_session(t1, session_id)
        await taking_service._require_session(t2, session_id)
        await taking_service.submit_session(t1, session_id, actor, arq_pool=_RecordingPool())
        await taking_service.submit_session(t2, session_id, actor, arq_pool=_RecordingPool())
    finally:
        await t1.close()
        await t2.close()

    assert len(enqueued) == 1, (
        f"exactly one caller may terminalize and enqueue, got {len(enqueued)}"
    )
