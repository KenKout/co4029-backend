"""Integration tests for SR card-review service (T7.5.5).

Covers plan §7.5.5:
* Atomic compose: derive_q + update_ef + scheduler + persist
* CardReview audit row written; StudentCardState upserted in same txn
* Failure path resets n=0, sets cooldown
* Calibration dampening for n <= 3
* Cross-feature ``quiz_questions`` read via ``text()`` (no QuizQuestion import)
* CardFailedEvent emitted on q==0
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.audit import current_actor_var, register_audit_listener
from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.identity import (
    models as _identity_models,  # noqa: F401  -- register users table for FK resolution
)
from abridgeai.features.quizzes import (
    models as _quiz_models,  # noqa: F401  -- register quiz_questions for FK resolution
)
from abridgeai.features.spaced_repetition.models import CardReview, StudentCardState
from abridgeai.features.spaced_repetition.services import (
    CardReviewResult,
    record_card_review,
)

register_audit_listener()


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def _seed_quiz(
    engine: AsyncEngine,
    *,
    t_exp_ms: int | None = 30000,
) -> tuple[UUID, UUID, UUID]:
    """Seed organization → user → course → module → quiz → question.

    Returns ``(student_id, question_id, quiz_attempt_id)``.
    """
    org_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_id = uuid.uuid4()
    attempt_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"sr-{org_id.hex[:8]}", "name": "SR T7.5.5"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"sr-{student_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": student_id,
                "slug": f"course-{course_id.hex[:8]}",
                "title": "T7.5.5 Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, :pos)"
            ),
            {"id": module_id, "course": course_id, "title": "M", "pos": 1},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :module, :title, 'draft')"
            ),
            {
                "id": quiz_id,
                "course": course_id,
                "module": module_id,
                "title": "T7.5.5 Quiz",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs, review_status"
                ") VALUES ("
                ":id, :quiz, 1, 'multiple_choice', 'Q?', "
                ":t_exp, '[]'::jsonb, 'pending'"
                ")"
            ),
            {"id": question_id, "quiz": quiz_id, "t_exp": t_exp_ms},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts (id, quiz_id, student_id, attempt_number) "
                "VALUES (:id, :quiz, :student, 1)"
            ),
            {"id": attempt_id, "quiz": quiz_id, "student": student_id},
        )
    return student_id, question_id, attempt_id


@pytest.mark.asyncio
async def test_first_review_correct_q4_creates_state(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    actor_token = current_actor_var.set(student_id)
    try:
        async with session_factory() as session, session.begin():
            result = await record_card_review(
                session,
                student_id=student_id,
                question_id=question_id,
                quiz_attempt_id=attempt_id,
                t_actual_ms=20000,
                correct=True,
                hint_used=False,
            )
    finally:
        current_actor_var.reset(actor_token)

    assert isinstance(result, CardReviewResult)
    assert result.q == 4
    assert result.passing is True
    assert result.repetition_count_after == 1
    assert result.interval_after == 1
    assert result.calibration_active is True
    assert result.retry_available_at is None
    assert result.due_at > datetime.now(tz=UTC)

    async with session_factory() as session:
        state = (
            await session.execute(
                select(StudentCardState).where(
                    StudentCardState.student_id == student_id,
                    StudentCardState.question_id == question_id,
                )
            )
        ).scalar_one()
        assert state.repetition_count == 1
        assert state.last_q == 4
        assert state.calibration_active is True
        assert state.total_reviews == 1

        review_rows = (
            (await session.execute(select(CardReview).where(CardReview.student_id == student_id)))
            .scalars()
            .all()
        )
        assert len(review_rows) == 1
        assert review_rows[0].q_derived == 4
        assert review_rows[0].n_before == 0
        assert review_rows[0].n_after == 1


@pytest.mark.asyncio
async def test_first_review_correct_q5_calibrated(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    async with session_factory() as session, session.begin():
        result = await record_card_review(
            session,
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=attempt_id,
            t_actual_ms=5000,  # rho=0.166 -> q=5
            correct=True,
            hint_used=False,
        )
    assert result.q == 5
    assert result.repetition_count_after == 1
    expected_ef_after = 2.5 + 0.6 * 0.1
    assert abs(result.ef_after - expected_ef_after) < 1e-6
    assert result.calibration_active is True


@pytest.mark.asyncio
async def test_failure_resets_and_sets_cooldown(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    async with session_factory() as session, session.begin():
        await record_card_review(
            session,
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=attempt_id,
            t_actual_ms=20000,
            correct=True,
            hint_used=False,
        )

    before_fail = datetime.now(tz=UTC)
    async with session_factory() as session, session.begin():
        result = await record_card_review(
            session,
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=attempt_id,
            t_actual_ms=99999,
            correct=False,
            hint_used=False,
        )
    after_fail = datetime.now(tz=UTC)

    assert result.q == 0
    assert result.passing is False
    assert result.repetition_count_after == 0
    assert result.interval_after == 1
    assert result.ef_after < result.ef_before
    assert result.retry_available_at is not None
    assert result.due_at == result.retry_available_at
    cooldown = timedelta(seconds=86400)
    assert before_fail + cooldown - timedelta(seconds=5) <= result.due_at
    assert result.due_at <= after_fail + cooldown + timedelta(seconds=5)


@pytest.mark.asyncio
async def test_atomicity_on_exception(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)

    with (
        patch(
            "abridgeai.features.spaced_repetition.services.review.derive_q",
            side_effect=RuntimeError("simulated failure inside service"),
        ),
        pytest.raises(RuntimeError, match="simulated failure"),
    ):
        async with session_factory() as session, session.begin():
            await record_card_review(
                session,
                student_id=student_id,
                question_id=question_id,
                quiz_attempt_id=attempt_id,
                t_actual_ms=99999,
                correct=False,
                hint_used=False,
            )

    async with session_factory() as session:
        states = (
            (
                await session.execute(
                    select(StudentCardState).where(StudentCardState.student_id == student_id)
                )
            )
            .scalars()
            .all()
        )
        reviews = (
            (await session.execute(select(CardReview).where(CardReview.student_id == student_id)))
            .scalars()
            .all()
        )
        assert states == []
        assert reviews == []


@pytest.mark.asyncio
async def test_no_t_exp_raises_value_error(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine, t_exp_ms=None)
    with pytest.raises(ValueError, match="expected_response_time_ms"):
        async with session_factory() as session, session.begin():
            await record_card_review(
                session,
                student_id=student_id,
                question_id=question_id,
                quiz_attempt_id=attempt_id,
                t_actual_ms=20000,
                correct=True,
                hint_used=False,
            )


@pytest.mark.asyncio
async def test_quiz_question_not_found_raises(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _question_id, _attempt_id = await _seed_quiz(engine)
    bogus = uuid.uuid4()
    with pytest.raises(NotFoundError, match=str(bogus)):
        async with session_factory() as session, session.begin():
            await record_card_review(
                session,
                student_id=student_id,
                question_id=bogus,
                quiz_attempt_id=None,
                t_actual_ms=20000,
                correct=True,
                hint_used=False,
            )


@pytest.mark.asyncio
async def test_card_review_audit_columns(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    async with session_factory() as session, session.begin():
        await record_card_review(
            session,
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=attempt_id,
            t_actual_ms=20000,
            correct=True,
            hint_used=False,
        )
    async with session_factory() as session:
        review = (
            await session.execute(select(CardReview).where(CardReview.student_id == student_id))
        ).scalar_one()
        assert review.created_at is not None
        assert review.t_actual_ms == 20000
        assert review.t_exp_ms == 30000
        assert review.correct is True


@pytest.mark.asyncio
async def test_subsequent_review_increments_n(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    intervals: list[int] = []
    for _ in range(3):
        async with session_factory() as session, session.begin():
            result = await record_card_review(
                session,
                student_id=student_id,
                question_id=question_id,
                quiz_attempt_id=attempt_id,
                t_actual_ms=20000,
                correct=True,
                hint_used=False,
            )
            intervals.append(result.interval_after)

    async with session_factory() as session:
        state = (
            await session.execute(
                select(StudentCardState).where(
                    StudentCardState.student_id == student_id,
                    StudentCardState.question_id == question_id,
                )
            )
        ).scalar_one()
        assert state.repetition_count == 3
        assert state.total_reviews == 3
    # SM-2 progression: n=0 -> 1 day; n=1 -> 6 days; n=2 -> ~round(6*ef)
    assert intervals[0] == 1
    assert 5 <= intervals[1] <= 7
    assert intervals[2] >= 13


@pytest.mark.asyncio
async def test_card_failed_event_queued_on_q0(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    async with session_factory() as session, session.begin():
        result = await record_card_review(
            session,
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=attempt_id,
            t_actual_ms=20000,
            correct=False,
            hint_used=False,
        )
    assert len(result.pending_events) == 1
    event = result.pending_events[0]
    assert event.student_id == student_id
    assert event.question_id == question_id
    assert event.quiz_attempt_id == attempt_id


@pytest.mark.asyncio
async def test_no_event_queued_on_passing(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, question_id, attempt_id = await _seed_quiz(engine)
    async with session_factory() as session, session.begin():
        result = await record_card_review(
            session,
            student_id=student_id,
            question_id=question_id,
            quiz_attempt_id=attempt_id,
            t_actual_ms=20000,
            correct=True,
            hint_used=False,
        )
    assert result.pending_events == []
