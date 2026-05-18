"""Unit tests for ``features.spaced_repetition.api.public`` (T25).

DB-backed: SR's public surface is a small consumer-facing API. Tests
seed minimal scaffolding, exercise each function, and roll back so the
shared DB stays clean.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.spaced_repetition.api import public
from abridgeai.features.spaced_repetition.api._dto import CardStateDTO
from abridgeai.features.spaced_repetition.api.public import (
    get_card_state,
    get_compliance_rate,
    get_due_card_count,
)


def test_module_exports() -> None:
    assert {
        "CardStateDTO",
        "get_card_state",
        "get_compliance_rate",
        "get_due_card_count",
    } <= set(public.__all__)


def test_card_state_dto_is_frozen() -> None:
    dto = CardStateDTO(
        student_id=uuid4(),
        question_id=uuid4(),
        ef=Decimal("2.500"),
        interval_days=1,
        repetition_count=0,
        due_at=datetime.now(UTC),
        last_q=None,
        last_reviewed_at=None,
        calibration_active=True,
        total_reviews=0,
    )
    with pytest.raises((TypeError, ValueError)):
        dto.interval_days = 99  # type: ignore[misc]


async def _seed_sr_scaffold(
    session: AsyncSession,
    *,
    student_id: UUID,
    org_id: UUID,
    owner_id: UUID,
    course_id: UUID,
    module_id: UUID,
    quiz_id: UUID,
) -> None:
    suffix = student_id.hex[:8]
    await session.execute(
        text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
        {"id": str(org_id), "slug": f"sr-{suffix}", "name": "SR API"},
    )
    await session.execute(
        text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
        {"id": str(owner_id), "email": f"owner-{suffix}@test.local"},
    )
    await session.execute(
        text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
        {"id": str(student_id), "email": f"student-{suffix}@test.local"},
    )
    await session.execute(
        text(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, "
            "title, status) VALUES (:id, :org, :owner, :slug, 'C', 'draft')"
        ),
        {"id": str(course_id), "org": str(org_id), "owner": str(owner_id), "slug": f"c-{suffix}"},
    )
    await session.execute(
        text(
            "INSERT INTO modules (id, course_id, title, position, status) "
            "VALUES (:id, :course, 'M', 1, 'draft')"
        ),
        {"id": str(module_id), "course": str(course_id)},
    )
    await session.execute(
        text(
            "INSERT INTO quizzes (id, course_id, module_id, title, status, "
            "passing_score_percent) "
            "VALUES (:id, :course, :module, 'Q', 'draft', 70.00)"
        ),
        {"id": str(quiz_id), "course": str(course_id), "module": str(module_id)},
    )


async def _seed_question(
    session: AsyncSession, *, question_id: UUID, quiz_id: UUID, position: int
) -> None:
    await session.execute(
        text(
            "INSERT INTO quiz_questions ("
            "id, quiz_id, position, question_type, prompt_text, "
            "expected_response_time_ms, source_refs, review_status"
            ") VALUES ("
            ":id, :quiz, :pos, 'multiple_choice', 'Q?', "
            "30000, '[]'::jsonb, 'approved')"
        ),
        {"id": str(question_id), "quiz": str(quiz_id), "pos": position},
    )


async def _seed_card(
    session: AsyncSession,
    *,
    student_id: UUID,
    question_id: UUID,
    due_at: datetime,
    ef: Decimal = Decimal("2.500"),
) -> None:
    await session.execute(
        text(
            "INSERT INTO student_card_state ("
            "student_id, question_id, ef, interval_days, repetition_count, "
            "due_at, total_reviews) VALUES ("
            ":sid, :qid, :ef, 1, 0, :due, 0)"
        ),
        {
            "sid": str(student_id),
            "qid": str(question_id),
            "ef": ef,
            "due": due_at,
        },
    )


async def test_get_due_card_count(test_engine: AsyncEngine) -> None:
    student_id = uuid4()
    org_id = uuid4()
    owner_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    quiz_id = uuid4()
    q_due_1 = uuid4()
    q_due_2 = uuid4()
    q_future = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_sr_scaffold(
                session,
                student_id=student_id,
                org_id=org_id,
                owner_id=owner_id,
                course_id=course_id,
                module_id=module_id,
                quiz_id=quiz_id,
            )
            for idx, qid in enumerate((q_due_1, q_due_2, q_future), start=1):
                await _seed_question(session, question_id=qid, quiz_id=quiz_id, position=idx)

            now = datetime.now(UTC)
            await _seed_card(
                session, student_id=student_id, question_id=q_due_1, due_at=now - timedelta(days=2)
            )
            await _seed_card(
                session, student_id=student_id, question_id=q_due_2, due_at=now - timedelta(hours=1)
            )
            await _seed_card(
                session,
                student_id=student_id,
                question_id=q_future,
                due_at=now + timedelta(days=3),
            )
            await session.flush()

            count = await get_due_card_count(session, student_id)
        finally:
            await trans.rollback()

    assert count == 2


async def test_get_card_state_found(test_engine: AsyncEngine) -> None:
    student_id = uuid4()
    org_id = uuid4()
    owner_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    quiz_id = uuid4()
    question_id = uuid4()
    due_at = datetime.now(UTC) + timedelta(days=1)

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_sr_scaffold(
                session,
                student_id=student_id,
                org_id=org_id,
                owner_id=owner_id,
                course_id=course_id,
                module_id=module_id,
                quiz_id=quiz_id,
            )
            await _seed_question(session, question_id=question_id, quiz_id=quiz_id, position=1)
            await _seed_card(
                session,
                student_id=student_id,
                question_id=question_id,
                due_at=due_at,
                ef=Decimal("2.700"),
            )
            await session.flush()

            state = await get_card_state(
                session, student_id=student_id, question_id=question_id
            )
        finally:
            await trans.rollback()

    assert state is not None
    assert isinstance(state, CardStateDTO)
    assert state.student_id == student_id
    assert state.question_id == question_id
    assert state.ef == Decimal("2.700")
    assert state.interval_days == 1
    assert state.repetition_count == 0
    assert state.calibration_active is True
    assert state.total_reviews == 0
    assert state.last_q is None


async def test_get_card_state_not_found(test_engine: AsyncEngine) -> None:
    async with AsyncSession(test_engine) as session:
        result = await get_card_state(
            session, student_id=uuid4(), question_id=uuid4()
        )
    assert result is None


async def test_get_compliance_rate_none_when_no_due_cards(test_engine: AsyncEngine) -> None:
    async with AsyncSession(test_engine) as session:
        rate = await get_compliance_rate(
            session, student_id=uuid4(), lesson_id=uuid4()
        )
    assert rate is None
