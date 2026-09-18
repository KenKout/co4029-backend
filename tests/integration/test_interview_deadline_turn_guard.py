"""P1 #5 regression: a timed session must refuse turns AT the deadline, not
only when the 5-minute sweep catches it.

The turn path checked only onboarding + status. Between ``deadline`` and the
next sweep, ``POST /respond`` still recorded and graded a late answer — the
time limit was not server-authoritative. ``take_session_step`` now enforces
the deadline itself with the SAME conditional terminalization the sweep uses
(so the closing reason and evaluation dispatch are identical), and refuses
the turn.
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
async def deadline_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """Session whose config carries time_limit_minutes=30; the test moves
    ``assessment_started_at`` to control expiry."""
    org_id = uuid4()
    teacher_id = uuid4()
    student_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    config_id = uuid4()
    question_id = uuid4()
    session_id = uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"dl-{suffix}", "name": "Deadline Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"dl-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'DL Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"dl-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'DL Interview', 'published', 'neutral', "
                ":teacher, :slug, 30)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"dl-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, prompt_text, position, review_status, question_type) "
                "VALUES (:id, :cfg, 'Describe a failure.', 1, 'approved', 'behavioral')"
            ),
            {"id": question_id, "cfg": config_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'hybrid', now(), now(), "
                "'completed', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {
        "config_id": config_id,
        "session_id": session_id,
        "student_id": student_id,
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
            text("DELETE FROM interview_questions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id}
        )
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"),
            {"t": teacher_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _expire(db: Any, probe: dict[str, Any], *, minutes_ago: int) -> None:
    from sqlalchemy import update

    from abridgeai.features.interviews.models import InterviewSession

    await db.execute(
        update(InterviewSession)
        .where(InterviewSession.id == probe["session_id"])
        .values(
            assessment_started_at=datetime.now(UTC) - timedelta(minutes=30 + minutes_ago)
        )
    )
    await db.commit()


async def test_a_turn_past_the_deadline_terminalizes_and_refuses(
    deadline_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Deadline passed, sweep has not run: the turn must terminalize the
    session (same conditional helper the sweep uses) and refuse the answer —
    no message row may appear."""
    from sqlalchemy import select

    from abridgeai.core.exceptions import AppError
    from abridgeai.features.interviews.models import (
        InterviewSession,
        InterviewSessionMessage,
    )
    from abridgeai.features.interviews.queries.sessions import count_terminal_sessions
    from abridgeai.features.interviews.services import taking
    from abridgeai.features.interviews.services.retake import (
        _RETAKE_CONSUMING_SESSION_STATUSES,
    )

    maker = _maker(test_engine)
    actor = type("_A", (), {"user_id": deadline_probe["student_id"]})()

    async with maker() as db:
        await _expire(db, deadline_probe, minutes_ago=2)
        with pytest.raises(AppError):
            await taking.take_session_step(
                db, deadline_probe["session_id"], "late answer", actor
            )

    # The session is terminal (timed_out: a gradeable turn existed? none yet —
    # abandoned is the honest reason for zero user turns) and accepts no rows.
    async with maker() as db:
        session = await db.get(InterviewSession, deadline_probe["session_id"])
        assert session is not None
        assert session.status in ("timed_out", "abandoned")
        assert session.status == "abandoned", "no user turn existed → abandoned"
        assert session.ended_at is not None
        rows = (
            (
                await db.execute(
                    select(InterviewSessionMessage).where(
                        InterviewSessionMessage.session_id == deadline_probe["session_id"],
                        InterviewSessionMessage.role == "user",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert rows == [], "the late answer must not be recorded"
        used = await count_terminal_sessions(
            db,
            deadline_probe["student_id"],
            deadline_probe["config_id"],
            _RETAKE_CONSUMING_SESSION_STATUSES,
        )
        assert used == 1, "the sweep and the turn path agree on consumption"


async def test_a_turn_before_the_deadline_still_lands(
    deadline_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Sanity: inside the window the guard is a no-op (the session stays live
    through the deadline check; the fold itself is out of scope here — the
    orchestrator raises on a missing pipeline, which proves the guard passed)."""
    from abridgeai.features.interviews.services import taking

    maker = _maker(test_engine)
    actor = type("_A", (), {"user_id": deadline_probe["student_id"]})()

    async with maker() as db:
        # assessment_started_at = now → 30 minutes of runway left.
        with pytest.raises(Exception) as exc_info:  # noqa: B017, PT011 - any pipeline error proves the deadline guard passed
            await taking.take_session_step(
                db, deadline_probe["session_id"], "late answer", actor
            )
        assert "deadline" not in str(exc_info.value).lower()


def _maker(test_engine: AsyncEngine) -> Any:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
