"""Integration tests for ``features.quizzes.services.taking`` (T5.13).

Plan §5398 security invariant: ``start_attempt`` must NOT leak
``is_correct`` (or any other answer-correctness field) on the take
payload. The taking service serializes through ``QuizQuestionPublic``
which intentionally drops the field; this test asserts the contract.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tests.support.db_graph import hard_delete_graph

import abridgeai.features.access_control.models  # noqa: F401  -- register users/orgs FK targets
import abridgeai.features.courses.models  # noqa: F401  -- register courses/modules/lessons FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- T6.1 registers interview_* tables
import abridgeai.features.spaced_repetition.models  # noqa: F401  -- register student_card_state
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.services import taking as taking_service

for _stub_name in ("interview_configs", "learning_materials", "learning_material_versions"):
    if _stub_name not in Base.metadata.tables:
        Table(
            _stub_name,
            Base.metadata,
            Column("id", PGUUID(as_uuid=True), primary_key=True),
        )


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
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def published_quiz(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qt-{suffix}", "name": "Take Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qt-owner-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"qt-student-{suffix}@test.local"},
        )
        # The student must belong to the quiz's organization: start_attempt now
        # enforces tenancy (organizations do not share quizzes).
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, status) "
                "VALUES (uuid_generate_v4(), :uid, :org, 'active')"
            ),
            {"uid": student_id, "org": org_id},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Take Course', 'published')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"course-{suffix}",
            },
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
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent, initial_ef) VALUES (:id, :course, :module, "
                "'Take Quiz', 'published', 70.00, 2.20)"
            ),
            {"id": quiz_id, "course": course_id, "module": module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status, "
                "expected_response_time_ms) "
                "VALUES (:id, :quiz, 1, 'multiple_choice', 'What is X?', 'approved', 30000)"
            ),
            {"id": question_id, "quiz": quiz_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_question_options "
                "(id, question_id, option_key, option_text, is_correct, position) "
                "VALUES (uuid_generate_v4(), :q, 'A', 'right', TRUE, 1), "
                "       (uuid_generate_v4(), :q, 'B', 'wrong', FALSE, 2), "
                "       (uuid_generate_v4(), :q, 'C', 'wrong', FALSE, 3), "
                "       (uuid_generate_v4(), :q, 'D', 'wrong', FALSE, 4)"
            ),
            {"q": question_id},
        )

    yield {
        "owner_id": owner_id,
        "student_id": student_id,
        "org_id": org_id,
        "course_id": course_id,
        "module_id": module_id,
        "quiz_id": quiz_id,
        "question_id": question_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM card_reviews WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id = :q)"
            ),
            {"q": quiz_id},
        )
        await conn.execute(
            text(
                "DELETE FROM student_card_state WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id = :q)"
            ),
            {"q": quiz_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_attempt_answers WHERE attempt_id IN "
                "(SELECT id FROM quiz_attempts WHERE quiz_id = :q)"
            ),
            {"q": quiz_id},
        )
        await conn.execute(
            text("DELETE FROM quiz_attempts WHERE quiz_id = :q"),
            {"q": quiz_id},
        )
        await conn.execute(
            text("DELETE FROM quiz_question_options WHERE question_id = :q"),
            {"q": question_id},
        )
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = :q"),
            {"q": quiz_id},
        )
        # Graph-driven: grading leaves quiz_grades under this quiz.
        await hard_delete_graph(conn, "quizzes", [str(quiz_id)])
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        # Graph-driven: enrollment/grade rows other suites attach to this
        # course block a bare delete (NO ACTION FKs).
        await hard_delete_graph(conn, "courses", [str(course_id)])
        await conn.execute(
            text("DELETE FROM organization_memberships WHERE user_id IN (:o, :s)"),
            {"o": owner_id, "s": student_id},
        )
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:o, :s)"),
            {"o": owner_id, "s": student_id},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_start_attempt_does_not_leak_is_correct(
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    async with session_factory() as session, session.begin():
        attempt, payload = await taking_service.start_attempt(
            session,
            published_quiz["quiz_id"],
            _actor(published_quiz["student_id"]),
        )
        await session.flush()

    assert attempt.attempt_number == 1
    serialized = payload.model_dump()
    questions = serialized["take"]["questions"]
    assert questions, "expected at least one question"
    for question in questions:
        assert "is_correct" not in question
        assert "correct_option_id" not in question
        for option in question["options"]:
            assert "is_correct" not in option, f"is_correct leaked into option payload: {option!r}"


@pytest.mark.asyncio
async def test_submit_attempt_grades_and_sets_passed(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    correct_option_id = (
        await (await engine.connect()).execute(
            text(
                "SELECT id FROM quiz_question_options WHERE question_id = :q AND is_correct = TRUE"
            ),
            {"q": published_quiz["question_id"]},
        )
    ).scalar_one()

    async with session_factory() as session, session.begin():
        attempt, _ = await taking_service.start_attempt(
            session,
            published_quiz["quiz_id"],
            _actor(published_quiz["student_id"]),
        )
        await session.flush()
        attempt_id = attempt.id

        class _Payload:
            question_id = published_quiz["question_id"]
            selected_option_id = correct_option_id
            answer_text = None
            t_actual_ms = 1234
            hint_used = False

        await taking_service.answer_attempt(
            session, attempt_id, _Payload(), _actor(published_quiz["student_id"])
        )
        await session.flush()
        graded = await taking_service.submit_attempt(
            session, attempt_id, _actor(published_quiz["student_id"])
        )
        await session.flush()

    assert graded.status == "submitted"
    assert graded.score_percent is not None
    assert float(graded.score_percent) == pytest.approx(100.0)
    assert graded.passed is True


async def _fetch_option_id(engine: AsyncEngine, question_id: uuid.UUID, *, correct: bool) -> uuid.UUID:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT id FROM quiz_question_options "
                    "WHERE question_id = :q AND is_correct = :c LIMIT 1"
                ),
                {"q": question_id, "c": correct},
            )
        ).scalar_one()


class _AnswerPayload:
    def __init__(
        self,
        question_id: uuid.UUID,
        selected_option_id: uuid.UUID | None = None,
        answer_text: str | None = None,
        t_actual_ms: int | None = None,
        hint_used: bool = False,
    ) -> None:
        self.question_id = question_id
        self.selected_option_id = selected_option_id
        self.answer_text = answer_text
        self.t_actual_ms = t_actual_ms
        self.hint_used = hint_used


@pytest.mark.asyncio
async def test_answer_attempt_fires_sm2_review_with_initial_ef(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    """FR-4.4: answering writes StudentCardState + CardReview in-txn;
    FR-4.2: the new card's EF seeds from ``Quiz.initial_ef`` (2.20)."""
    student_id = published_quiz["student_id"]
    question_id = published_quiz["question_id"]
    correct_option = await _fetch_option_id(engine, question_id, correct=True)

    async with session_factory() as session, session.begin():
        attempt, _ = await taking_service.start_attempt(
            session, published_quiz["quiz_id"], _actor(student_id)
        )
        await session.flush()
        # rho = 20000/30000 = 0.667 -> q=4 -> EF delta 0 (calibrated)
        _, review = await taking_service.answer_attempt(
            session,
            attempt.id,
            _AnswerPayload(question_id, selected_option_id=correct_option, t_actual_ms=20000),
            _actor(student_id),
        )

    assert review is not None
    assert review.q == 4
    assert review.ef_before == pytest.approx(2.20)
    assert review.repetition_count_after == 1
    assert review.pending_events == []

    async with engine.connect() as conn:
        state_row = (
            await conn.execute(
                text(
                    "SELECT ef, repetition_count, total_reviews FROM student_card_state "
                    "WHERE student_id = :s AND question_id = :q"
                ),
                {"s": student_id, "q": question_id},
            )
        ).one()
        assert float(state_row.ef) == pytest.approx(2.20)
        assert state_row.repetition_count == 1
        assert state_row.total_reviews == 1

        review_count = (
            await conn.execute(
                text("SELECT COUNT(*) FROM card_reviews WHERE student_id = :s"),
                {"s": student_id},
            )
        ).scalar_one()
        assert review_count == 1


@pytest.mark.asyncio
async def test_answer_attempt_failure_sets_cooldown_and_pending_event(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    """Incorrect answer -> q=0: counters reset, ~24h cooldown, and a
    CardFailedEvent queued for post-commit remediation dispatch."""
    student_id = published_quiz["student_id"]
    question_id = published_quiz["question_id"]
    wrong_option = await _fetch_option_id(engine, question_id, correct=False)

    before = datetime.now(tz=UTC)
    async with session_factory() as session, session.begin():
        attempt, _ = await taking_service.start_attempt(
            session, published_quiz["quiz_id"], _actor(student_id)
        )
        await session.flush()
        _, review = await taking_service.answer_attempt(
            session,
            attempt.id,
            _AnswerPayload(question_id, selected_option_id=wrong_option, t_actual_ms=20000),
            _actor(student_id),
        )

    assert review is not None
    assert review.q == 0
    assert review.passing is False
    assert review.repetition_count_after == 0
    assert review.retry_available_at is not None
    assert review.retry_available_at >= before + timedelta(hours=23)
    assert len(review.pending_events) == 1
    event = review.pending_events[0]
    assert event.student_id == student_id
    assert event.question_id == question_id
    assert event.quiz_id == published_quiz["quiz_id"]


@pytest.mark.asyncio
async def test_answer_attempt_skips_sr_without_t_exp(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    """A question without T_exp (draft-era data) never blocks the answer
    write — SR review is skipped and the answer persists."""
    student_id = published_quiz["student_id"]
    no_t_exp_question = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status) "
                "VALUES (:id, :quiz, 2, 'short_answer', 'Describe X.', 'approved')"
            ),
            {"id": no_t_exp_question, "quiz": published_quiz["quiz_id"]},
        )

    async with session_factory() as session, session.begin():
        attempt, _ = await taking_service.start_attempt(
            session, published_quiz["quiz_id"], _actor(student_id)
        )
        await session.flush()
        answer, review = await taking_service.answer_attempt(
            session,
            attempt.id,
            _AnswerPayload(no_t_exp_question, answer_text="an answer", t_actual_ms=5000),
            _actor(student_id),
        )

    assert review is None
    assert answer.id is not None

    async with engine.connect() as conn:
        state_count = (
            await conn.execute(
                text("SELECT COUNT(*) FROM student_card_state WHERE question_id = :q"),
                {"q": no_t_exp_question},
            )
        ).scalar_one()
        assert state_count == 0


@pytest.mark.asyncio
async def test_start_attempt_rejects_draft_quiz(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    """Students cannot start attempts on unpublished quizzes (FR-4.3
    hardening: the take surface now loads via the published query)."""
    draft_quiz_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES (:id, :course, :module, "
                "'Draft Quiz', 'draft', 70.00)"
            ),
            {
                "id": draft_quiz_id,
                "course": published_quiz["course_id"],
                "module": published_quiz["module_id"],
            },
        )

    try:
        with pytest.raises(NotFoundError):
            async with session_factory() as session, session.begin():
                await taking_service.start_attempt(
                    session, draft_quiz_id, _actor(published_quiz["student_id"])
                )
    finally:
        async with engine.begin() as conn:
            await hard_delete_graph(conn, "quizzes", [str(draft_quiz_id)])


@pytest.mark.asyncio
async def test_start_attempt_rejects_cross_org_student(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    """Organizations do not share quizzes: a student who is NOT a member of the
    quiz's organization cannot start an attempt, even on a published quiz.

    Raises NotFoundError (router -> 404), the same shape a missing quiz returns,
    so the endpoint cannot be used to probe which quiz ids exist in another
    tenant. The outsider has a valid account (and even a membership in a
    DIFFERENT org) — only the tenancy mismatch blocks them.
    """
    outsider_id = uuid.uuid4()
    other_org = uuid.uuid4()
    suffix = outsider_id.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": other_org, "slug": f"outsider-org-{suffix}", "name": "Outsider Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": outsider_id, "email": f"outsider-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, status) "
                "VALUES (uuid_generate_v4(), :uid, :org, 'active')"
            ),
            {"uid": outsider_id, "org": other_org},
        )

    try:
        with pytest.raises(NotFoundError):
            async with session_factory() as session, session.begin():
                await taking_service.start_attempt(
                    session, published_quiz["quiz_id"], _actor(outsider_id)
                )
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM organization_memberships WHERE user_id = :uid"),
                {"uid": outsider_id},
            )
            await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": outsider_id})
            await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": other_org})


@pytest.mark.asyncio
async def test_shuffled_take_payload_reflects_layout_and_survives_position_sort(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    published_quiz: dict,
) -> None:
    """A shuffled attempt's take payload must present the SHUFFLED order.

    Regression for the anti-cheat leak: apply_layout reorders the questions
    but the SPA re-sorts by ``position``. The take payload must therefore
    carry sequential display positions (1..N) matching the persisted layout
    order, so sorting by position preserves the shuffle instead of restoring
    the authored order. Verified for both the start payload and the resume
    (get_attempt_progress) payload.
    """
    quiz_id = published_quiz["quiz_id"]
    student_id = published_quiz["student_id"]
    # Turn shuffle on and add enough questions that a shuffle is near-certain
    # to differ from authored order (10 questions → 1/10! chance of identity).
    extra_ids = [uuid.uuid4() for _ in range(9)]
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE quizzes SET shuffle_questions = TRUE WHERE id = :id"),
            {"id": quiz_id},
        )
        for i, qid in enumerate(extra_ids, start=2):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, review_status, "
                    "expected_response_time_ms) "
                    "VALUES (:id, :quiz, :pos, 'multiple_choice', :prompt, 'approved', 30000)"
                ),
                {"id": qid, "quiz": quiz_id, "pos": i, "prompt": f"Q{i}?"},
            )

    try:
        async with session_factory() as session, session.begin():
            attempt, payload = await taking_service.start_attempt(
                session, quiz_id, _actor(student_id)
            )
            attempt_id = attempt.id
            layout = attempt.layout
            start_qids = [str(q.id) for q in payload.take.questions]
            start_positions = [q.position for q in payload.take.questions]

        # Positions are sequential 1..N (display slots), NOT the authored order.
        assert start_positions == list(range(1, len(start_qids) + 1))
        # The take payload order matches the persisted layout question_order.
        assert layout is not None
        assert start_qids == layout["question_order"]
        # Sorting by position (as the SPA does) preserves the shuffled order.
        by_position = sorted(
            payload.take.questions, key=lambda q: q.position
        )
        assert [str(q.id) for q in by_position] == start_qids

        # Resume must reproduce the SAME shuffled order (re-reads layout).
        async with session_factory() as session, session.begin():
            resume = await taking_service.get_attempt_progress(
                session, attempt_id=attempt_id, actor=_actor(student_id)
            )
        assert resume is not None
        resume_qids = [str(q.id) for q in resume.take.questions]
        resume_positions = [q.position for q in resume.take.questions]
        assert resume_qids == start_qids
        assert resume_positions == list(range(1, len(resume_qids) + 1))
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "DELETE FROM quiz_attempt_answers WHERE attempt_id IN "
                    "(SELECT id FROM quiz_attempts WHERE quiz_id = :q)"
                ),
                {"q": quiz_id},
            )
            await conn.execute(
                text("DELETE FROM quiz_attempts WHERE quiz_id = :q"), {"q": quiz_id}
            )
            await conn.execute(
                text("DELETE FROM quiz_questions WHERE id = ANY(:ids)"),
                {"ids": extra_ids},
            )
            await conn.execute(
                text("UPDATE quizzes SET shuffle_questions = FALSE WHERE id = :id"),
                {"id": quiz_id},
            )


@pytest.mark.asyncio
async def test_initial_ef_out_of_range_is_clamped(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A teacher-set ``initial_ef`` outside [1.3, 2.5] must clamp, not 500.

    ``student_card_state.ef`` is CHECK-constrained to [1.3, 2.5]
    (``ck_student_card_state_ef_range``) and the quiz settings accept
    ``initial_ef`` without range validation, so an out-of-range value (e.g.
    3.0) would violate the constraint on the first review and blow up with
    an IntegrityError. ``_load_or_init_state`` clamps it into range instead.
    """
    org_id, owner_id, student_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    course_id, module_id, quiz_id, question_id = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qt-clamp-{suffix}", "name": "Clamp Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qt-clamp-owner-{suffix}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"qt-clamp-student-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, status) "
                "VALUES (uuid_generate_v4(), :uid, :org, 'active')"
            ),
            {"uid": student_id, "org": org_id},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Clamp Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": owner_id, "slug": f"cclamp-{suffix}"},
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
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent, initial_ef) VALUES (:id, :course, :module, "
                "'Clamp Quiz', 'published', 70.00, 3.00)"
            ),
            {"id": quiz_id, "course": course_id, "module": module_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text, review_status, "
                "expected_response_time_ms) "
                "VALUES (:id, :quiz, 1, 'multiple_choice', 'Clamp?', 'approved', 30000)"
            ),
            {"id": question_id, "quiz": quiz_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_question_options "
                "(id, question_id, option_key, option_text, is_correct, position) "
                "VALUES (uuid_generate_v4(), :q, 'A', 'right', TRUE, 1), "
                "       (uuid_generate_v4(), :q, 'B', 'wrong', FALSE, 2)"
            ),
            {"q": question_id},
        )

    try:
        correct_option = await _fetch_option_id(engine, question_id, correct=True)
        async with session_factory() as session, session.begin():
            attempt, _ = await taking_service.start_attempt(
                session, quiz_id, _actor(student_id)
            )
            await session.flush()
            _, review = await taking_service.answer_attempt(
                session,
                attempt.id,
                _AnswerPayload(
                    question_id,
                    selected_option_id=correct_option,
                    t_actual_ms=30000,
                ),
                _actor(student_id),
            )

        assert review is not None
        # initial_ef 3.0 clamps to the 2.5 ceiling (never raises IntegrityError).
        assert review.ef_before == pytest.approx(2.5)
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM card_reviews WHERE question_id = :q"), {"q": question_id}
            )
            await conn.execute(
                text("DELETE FROM student_card_state WHERE question_id = :q"), {"q": question_id}
            )
            await conn.execute(
                text(
                    "DELETE FROM quiz_attempt_answers WHERE attempt_id IN "
                    "(SELECT id FROM quiz_attempts WHERE quiz_id = :q)"
                ),
                {"q": quiz_id},
            )
            await conn.execute(
                text("DELETE FROM quiz_attempts WHERE quiz_id = :q"), {"q": quiz_id}
            )
            await conn.execute(
                text("DELETE FROM quiz_question_options WHERE question_id = :q"), {"q": question_id}
            )
            await conn.execute(text("DELETE FROM quiz_questions WHERE id = :q"), {"q": question_id})
            await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz_id})
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
            await conn.execute(
                text("DELETE FROM organization_memberships WHERE user_id = :u"), {"u": student_id}
            )
            await conn.execute(
                text("DELETE FROM users WHERE id IN (:o, :s)"), {"o": owner_id, "s": student_id}
            )
            await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})
