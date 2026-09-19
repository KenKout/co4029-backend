"""P1 #7 regression: the closing ceremony must match the PERSISTED terminal
status, never the loser's requested reason.

Race: the sweep (or hard-stop) terminalizes a session ``timed_out`` while a
finisher is in flight. The finisher loses the terminal CAS but its ceremony
was inserted BEFORE the CAS with ``reason="natural"`` — the session row says
``timed_out`` while the transcript says "that concludes your interview". The
reverse race (finisher wins ``completed``, sweep loses) left the same
contradiction when the sweep's timeout ceremony landed first.

Contract now: the ceremony is written only by the caller whose conditional
terminalization WON, and it is written AFTER the CAS from the WON status. A
loser (or an already-terminal refinish) must return the WINNER's persisted
closing, and never insert a second one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def finish_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    org_id = uuid4()
    teacher_id = uuid4()
    student_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    config_id = uuid4()
    session_id = uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"fr-{suffix}", "name": "Finish Org"},
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
                ":teacher, :slug, 30)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id,
             "slug": f"fr-config-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, onboarding_stage, interview_language, assessment_started_at) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'hybrid', now() - interval '1 hour', "
                "'completed', 'en', now() - interval '40 minutes')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, prompt_text, question_type, position) "
                "VALUES (:id, :cfg, 'Warm-up?', 'conceptual', 1)"
            ),
            {"id": uuid4(), "cfg": config_id},
        )
        sq_id = uuid4()
        await conn.execute(
            text(
                "INSERT INTO interview_session_questions "
                "(id, session_id, interview_question_id, sequence_no) "
                "VALUES (:id, :s, (SELECT id FROM interview_questions "
                "WHERE interview_config_id = :cfg LIMIT 1), 1)"
            ),
            {"id": sq_id, "s": session_id, "cfg": config_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_session_messages "
                "(id, session_id, session_question_id, role, content_text, metadata_json) "
                "VALUES (:id, :s, :sq, 'user', 'my answer', '{}')"
            ),
            {"id": uuid4(), "s": session_id, "sq": sq_id},
        )

    yield {"session_id": session_id, "student_id": student_id}

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM interview_session_messages WHERE session_id = :s"),
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


def _actor(student_id: Any) -> Any:
    return type("_A", (), {"user_id": student_id})()


async def _closing_rows(test_engine: AsyncEngine, session_id: Any) -> list[dict[str, Any]]:
    async with test_engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT content_text, metadata_json FROM interview_session_messages "
                    "WHERE session_id = :s AND metadata_json->>'ceremony_key' = 'closing'"
                ),
                {"s": session_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


async def test_loser_natural_finish_after_sweep_timed_out_writes_no_natural_closing(
    finish_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The sweep wins ``timed_out`` first; the candidate's finish arrives
    after. The transcript must carry NO 'that concludes / đến đây là kết thúc'
    natural closing — only the timeout closing."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import lifecycle as lifecycle_service
    from abridgeai.features.interviews.services import taking as taking_service

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    actor = _actor(finish_probe["student_id"])

    # The sweep terminalizes first (committed).
    async with maker() as db:
        await lifecycle_service.sweep_expired_interview_sessions(db)
        await db.commit()

    # Now the candidate's finish (reason=natural default) hits the terminal row.
    async with maker() as db:
        session = await taking_service.submit_session(
            db, finish_probe["session_id"], actor, arq_pool=None, reason="natural"
        )
        await db.commit()

    assert session.status == "timed_out"
    closings = await _closing_rows(test_engine, finish_probe["session_id"])
    assert len(closings) == 1, "exactly one closing ceremony"
    natural_markers = ("đến đây là kết thúc", "That concludes")
    assert not any(
        marker in closings[0]["content_text"] for marker in natural_markers
    ), f"natural closing on a timed_out session: {closings[0]['content_text']!r}"
    assert closings[0]["metadata_json"].get("finish_reason") == "timed_out"


async def test_sweep_loser_to_a_completed_finisher_writes_no_timeout_closing(
    finish_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Reverse order: the candidate's finish wins ``completed``; a stale sweep
    (arriving before the deadline math catches up, via finalize tolerance)
    must not stamp a timeout closing onto the completed transcript."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from abridgeai.features.interviews.services import lifecycle as lifecycle_service
    from abridgeai.features.interviews.services import taking as taking_service

    maker = async_sessionmaker(test_engine, expire_on_commit=False)
    actor = _actor(finish_probe["student_id"])
    # Candidate finish FIRST (wins completed), sweep second.
    async with maker() as db:
        await taking_service.submit_session(
            db, finish_probe["session_id"], actor, arq_pool=None, reason="natural"
        )
        await db.commit()
    async with maker() as db:
        await lifecycle_service.sweep_expired_interview_sessions(db)
        await db.commit()

    async with test_engine.begin() as conn:
        status_row = (
            await conn.execute(
                text("SELECT status FROM interview_sessions WHERE id = :s"),
                {"s": finish_probe["session_id"]},
            )
        ).scalar_one()
    assert status_row == "completed"
    closings = await _closing_rows(test_engine, finish_probe["session_id"])
    assert len(closings) == 1, "exactly one closing ceremony"
    timeout_markers = ("đã kết thúc", "time for your")
    assert not any(
        marker in closings[0]["content_text"] for marker in timeout_markers
    ), f"timeout closing on a completed session: {closings[0]['content_text']!r}"
    assert closings[0]["metadata_json"].get("finish_reason") == "natural"
