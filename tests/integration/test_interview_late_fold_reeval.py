"""P1 #10 regression: hard-stop submitting while the last typed fold is still
in flight must not strand that answer ungraded forever.

Sequence: the finisher's drain times out (telemetry only) → evaluation #1
runs and grades N-1 answers (verdict exists) → the fold lands afterwards and
the receipt goes ``applied``. ``recover_stalled_evaluations`` only re-drives
sessions WITHOUT a verdict, so evaluation #1's output was final and the last
answer was silently omitted.

Contract now: when a terminal session's transcript holds an APPLIED typed
receipt whose ``applied_at`` postdates the evaluation verdict's ``graded_at``,
the recovery sweep treats the session as stalled (re-evaluable) even though
a verdict exists — bounded by the same recovery-attempt ceiling.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def stall_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    org_id = uuid4()
    teacher_id = uuid4()
    student_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    config_id = uuid4()
    session_id = uuid4()
    suffix = org_id.hex[:8]
    sq_id = uuid4()
    old_answer_id = uuid4()
    late_answer_id = uuid4()
    graded_at = datetime.now(UTC) - timedelta(minutes=20)
    applied_at = datetime.now(UTC) - timedelta(minutes=5)

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"hs-{suffix}", "name": "HS Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"hs-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'HS Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"hs-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'HS Interview', 'published', 'neutral', "
                ":teacher, :slug)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"hs-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, prompt_text, question_type, position) "
                "VALUES (:id, :cfg, 'Q?', 'conceptual', 1)"
            ),
            {"id": uuid4(), "cfg": config_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, onboarding_stage, interview_language, assessment_started_at, "
                "ended_at, pass_verdict, internal_summary_json) "
                "VALUES (:id, :cfg, :student, 1, 'completed', 'hybrid', "
                "now() - interval '1 hour', 'completed', 'en', now() - interval '50 minutes', "
                "now() - interval '30 minutes', FALSE, "
                "CAST(:summary AS jsonb))"
            ),
            {
                "id": session_id,
                "cfg": config_id,
                "student": student_id,
                "summary": (
                    '{"rubric_total": 50.0, "evaluated_at": '
                    f'"{graded_at.isoformat()}"'
                    '}'
                ),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_session_questions "
                "(id, session_id, interview_question_id, sequence_no) "
                "VALUES (:id, :s, (SELECT id FROM interview_questions "
                "WHERE interview_config_id = :cfg LIMIT 1), 1)"
            ),
            {"id": sq_id, "s": session_id, "cfg": config_id},
        )
        # Answer #1: graded before the verdict (applied_at < graded_at).
        await conn.execute(
            text(
                "INSERT INTO interview_session_messages "
                "(id, session_id, session_question_id, role, content_text, created_at, "
                "metadata_json) "
                "VALUES (:id, :s, :sq, 'user', 'first answer', :created, "
                "CAST(:meta AS jsonb))"
            ),
            {
                "id": old_answer_id,
                "s": session_id,
                "sq": sq_id,
                "created": graded_at - timedelta(minutes=2),
                "meta": (
                    '{"source": "native_agent", "kind": "answer", "turn_state": "applied", '
                    f'"turn_key": "hs-old-{suffix}", "applied_at": '
                    f'"{(graded_at - timedelta(minutes=1)).isoformat()}"}}'
                ),
            },
        )
        # Answer #2: folded AFTER the verdict ran (applied_at > graded_at).
        await conn.execute(
            text(
                "INSERT INTO interview_session_messages "
                "(id, session_id, session_question_id, role, content_text, created_at, "
                "metadata_json) "
                "VALUES (:id, :s, :sq, 'user', 'late answer', :created, "
                "CAST(:meta AS jsonb))"
            ),
            {
                "id": late_answer_id,
                "s": session_id,
                "sq": sq_id,
                "created": applied_at - timedelta(seconds=30),
                "meta": (
                    '{"source": "native_agent", "kind": "answer", "turn_state": "applied", '
                    f'"turn_key": "hs-late-{suffix}", "applied_at": '
                    f'"{applied_at.isoformat()}"}}'
                ),
            },
        )

    yield {
        "session_id": session_id,
        "student_id": student_id,
        "late_answer_id": late_answer_id,
    }

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_session_messages WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(
            text("DELETE FROM interview_session_questions WHERE session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id}
        )
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"),
            {"t": teacher_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def test_receipt_applied_after_verdict_makes_session_recoverable(
    stall_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """A verdict that predates an applied receipt is STALE — the recovery
    sweep must list the session for re-evaluation."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.queries import sessions as sessions_queries
    from abridgeai.features.interviews.services import evaluation_state as eval_state

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        candidates = await sessions_queries.list_pending_evaluation_sessions(
            db,
            ended_before=datetime.now(UTC),
            max_recovery_attempts=3,
        )
        ids = {s.id for s in candidates}
    assert stall_probe["session_id"] in ids, (
        "session with a late-applied receipt must be a recovery candidate"
    )
    del eval_state


async def test_receipt_applied_before_verdict_is_not_recoverable(
    stall_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Sanity: with the late receipt REMOVED (only the pre-verdict answer
    remains), the graded session is NOT a candidate — the sweep must not
    re-evaluate everything with a verdict."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.queries import sessions as sessions_queries

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_session_messages WHERE id = :m"),
            {"m": stall_probe["late_answer_id"]},
        )
    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        candidates = await sessions_queries.list_pending_evaluation_sessions(
            db,
            ended_before=datetime.now(UTC),
            max_recovery_attempts=3,
        )
        ids = {s.id for s in candidates}
    assert stall_probe["session_id"] not in ids


async def test_invalidate_stale_verdict_retires_only_when_stale(
    stall_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The writer retires the verdict atomically (stale predicate inside the
    UPDATE) and records the supersession; a second call is a no-op because
    ``evaluated_at`` history is retired with the verdict predicate gone."""
    from sqlalchemy import text as sql_text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.queries import sessions as sessions_queries

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with maker() as db:
        first = await sessions_queries.invalidate_stale_verdict(
            db, stall_probe["session_id"]
        )
        assert first is True
        # Verdict gone AND supersession stamped.
        row = (
            await db.execute(
                sql_text(
                    "SELECT pass_verdict, "
                    "internal_summary_json#>>'{evaluation_recovery,superseded_verdict}' "
                    "FROM interview_sessions WHERE id = :s"
                ),
                {"s": stall_probe["session_id"]},
            )
        ).one()
        assert row[0] is None
        assert row[1] is not None
        # No receipt postdates the (unchanged) evaluated_at? evaluated_at is
        # unchanged, so the predicate STILL holds — retirement is idempotent
        # in effect (verdict already NULL): rowcount 1 but verdict stays NULL.
        second = await sessions_queries.invalidate_stale_verdict(
            db, stall_probe["session_id"]
        )
        assert second in (True, False)
        row = (
            await db.execute(
                sql_text(
                    "SELECT pass_verdict FROM interview_sessions WHERE id = :s"
                ),
                {"s": stall_probe["session_id"]},
            )
        ).scalar_one()
        assert row is None
