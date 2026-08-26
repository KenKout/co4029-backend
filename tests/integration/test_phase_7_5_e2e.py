"""Phase 7.5 closer end-to-end tests (T7.5.13).

The thesis full-flow scenario stitches the cognitive layer together:
manager-style enrollment → quiz publish → card review → remediation
hook → unlock gate → time-advance retry → unlock satisfied. The two
companion scenarios cover class-wide difficulty detection and the
compliance ratio surfaced in the at-risk view.

Mocks
-----
* ``datetime.now`` inside :mod:`abridgeai.features.spaced_repetition.services.review`
  is monkeypatched per scenario to simulate the 24h failure cooldown
  without sleeping.
* :func:`dispatch_remediation_for_card_failure` is patched to record
  invocations; the real KG path needs Neo4j and is exercised in
  T7.5.10.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.audit import register_audit_listener
from abridgeai.core.config import get_settings
from abridgeai.features.identity import models as _identity_models  # noqa: F401  -- FK targets
from abridgeai.features.quizzes import models as _quiz_models  # noqa: F401  -- FK targets
from abridgeai.features.spaced_repetition import models as _sr_models  # noqa: F401
from abridgeai.features.spaced_repetition.queries import (
    at_risk_students,
    class_card_difficulty,
    review_compliance_rate,
    student_lesson_summary,
)
from abridgeai.features.spaced_repetition.services import record_card_review
from abridgeai.features.spaced_repetition.sm2.lesson_unlock import (
    check_lesson_unlock,
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
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest.fixture(autouse=True)
def _isolate_cache() -> AsyncIterator[None]:
    """Bypass Redis so unlock + KR + compliance always re-compute from DB."""
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=None)
    fake_client.set = AsyncMock(return_value=None)
    with (
        patch("abridgeai.core.cache.decorators.get_cache", return_value=fake_client),
        patch(
            "abridgeai.features.spaced_repetition.sm2.lesson_unlock.get_cache",
            return_value=fake_client,
            create=True,
        ),
    ):
        yield


@pytest_asyncio.fixture
async def thesis_seed(engine: AsyncEngine) -> AsyncIterator[dict[str, object]]:
    """One organization + Alice + course + module + 2 lessons + 1 quiz with 5 questions.

    Lesson 1 unlocks at ``ef_min=2.0`` / ``tau=1.0`` (every card must be
    above floor) so a single failed card blocks the gate. Lesson 2
    keeps the same ef_min and depends on Lesson 1 via the prereq edge.
    """
    org_id = uuid.uuid4()
    alice_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    current_lesson_id = uuid.uuid4()
    next_lesson_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    qids = [uuid.uuid4() for _ in range(5)]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"e2e-{org_id.hex[:8]}", "name": "Networking 101 Co."},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": alice_id, "email": f"alice-{alice_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": alice_id,
                "slug": f"course-{course_id.hex[:8]}",
                "title": "Networking 101",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                "VALUES (:id, :course, :student, 'active')"
            ),
            {"id": uuid.uuid4(), "course": course_id, "student": alice_id},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, :title, 1, 'published')"
            ),
            {"id": module_id, "course": course_id, "title": "Module 1"},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status, ef_min_unlock, tau_unlock, "
                "requires_interview_pass) "
                "VALUES (:id, :m, :slug, :title, 'published', :ef, :tau, FALSE)"
            ),
            {
                "id": current_lesson_id,
                "m": module_id,
                "slug": f"l-cur-{current_lesson_id.hex[:6]}",
                "title": "Lesson 1: Subnetting",
                "ef": Decimal("2.0"),
                "tau": Decimal("1.0"),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status, ef_min_unlock, tau_unlock, "
                "requires_interview_pass) "
                "VALUES (:id, :m, :slug, :title, 'published', :ef, :tau, FALSE)"
            ),
            {
                "id": next_lesson_id,
                "m": module_id,
                "slug": f"l-nxt-{next_lesson_id.hex[:6]}",
                "title": "Lesson 2: Routing",
                "ef": Decimal("2.0"),
                "tau": Decimal("0.8"),
            },
        )
        await conn.execute(
            text("INSERT INTO lesson_prerequisites (lesson_id, prereq_lesson_id) VALUES (:l, :p)"),
            {"l": next_lesson_id, "p": current_lesson_id},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, slug) VALUES (:id, :course, :m, :title, 'published', 'slug-' || uuid_generate_v4()::text);"
            ),
            {"id": quiz_id, "course": course_id, "m": module_id, "title": "Quiz 1"},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items (id, module_id, item_type, quiz_id, position) "
                "VALUES (:id, :m, 'quiz', :q, 1)"
            ),
            {"id": uuid.uuid4(), "m": module_id, "q": quiz_id},
        )
        await conn.execute(
            text("INSERT INTO quiz_source_lessons (quiz_id, lesson_id) VALUES (:q, :l)"),
            {"q": quiz_id, "l": current_lesson_id},
        )
        for idx, qid in enumerate(qids, start=1):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions ("
                    "id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs, review_status) "
                    "VALUES ("
                    ":id, :quiz, :pos, 'multiple_choice', 'Q?', "
                    "30000, '[]'::jsonb, 'approved')"
                ),
                {"id": qid, "quiz": quiz_id, "pos": idx},
            )

    seed: dict[str, object] = {
        "org_id": org_id,
        "alice_id": alice_id,
        "course_id": course_id,
        "module_id": module_id,
        "current_lesson_id": current_lesson_id,
        "next_lesson_id": next_lesson_id,
        "quiz_id": quiz_id,
        "qids": qids,
    }
    try:
        yield seed
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM card_reviews WHERE student_id = :s"), {"s": alice_id}
            )
            await conn.execute(
                text("DELETE FROM student_card_state WHERE student_id = :s"), {"s": alice_id}
            )
            await conn.execute(text("DELETE FROM quiz_attempts WHERE quiz_id = :q"), {"q": quiz_id})
            await conn.execute(
                text("DELETE FROM quiz_questions WHERE quiz_id = :q"), {"q": quiz_id}
            )
            await conn.execute(
                text("DELETE FROM quiz_source_lessons WHERE quiz_id = :q"), {"q": quiz_id}
            )
            await conn.execute(text("DELETE FROM module_items WHERE quiz_id = :q"), {"q": quiz_id})
            await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz_id})
            await conn.execute(
                text("DELETE FROM lesson_prerequisites WHERE lesson_id = :l"),
                {"l": next_lesson_id},
            )
            await conn.execute(
                text("DELETE FROM lessons WHERE id IN (:a, :b)"),
                {"a": current_lesson_id, "b": next_lesson_id},
            )
            await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
            await conn.execute(
                text("DELETE FROM course_enrollments WHERE course_id = :c"),
                {"c": course_id},
            )
            await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": alice_id})
            await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _seed_quiz_attempt(engine: AsyncEngine, *, quiz_id: UUID, student_id: UUID) -> UUID:
    attempt_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts "
                "(id, quiz_id, student_id, attempt_number, status, score_percent) "
                "VALUES (:id, :quiz, :student, 1, 'graded', :score)"
            ),
            {
                "id": attempt_id,
                "quiz": quiz_id,
                "student": student_id,
                "score": Decimal("80"),
            },
        )
    return attempt_id


@pytest.mark.asyncio
async def test_thesis_full_flow(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    thesis_seed: dict[str, object],
) -> None:
    """End-to-end: enrol → quiz attempt → review → remediation → unlock gate.

    Asserts the headline thesis flow holds: a single failed card blocks
    progression, the calibrated bump after the cooldown lets ef recover,
    and once it crosses ``ef_min_unlock`` the gate flips to eligible.
    """
    alice_id = UUID(str(thesis_seed["alice_id"]))
    course_id = UUID(str(thesis_seed["course_id"]))
    current_lesson_id = UUID(str(thesis_seed["current_lesson_id"]))
    next_lesson_id = UUID(str(thesis_seed["next_lesson_id"]))
    quiz_id = UUID(str(thesis_seed["quiz_id"]))
    qids = [UUID(str(q)) for q in thesis_seed["qids"]]  # type: ignore[union-attr]

    attempt_id = await _seed_quiz_attempt(engine, quiz_id=quiz_id, student_id=alice_id)

    dispatcher = AsyncMock()
    with patch(
        "abridgeai.features.spaced_repetition.services.remediation."
        "dispatch_remediation_for_card_failure",
        new=dispatcher,
    ):
        for qid in qids[:4]:
            async with session_factory() as session, session.begin():
                result = await record_card_review(
                    session,
                    student_id=alice_id,
                    question_id=qid,
                    quiz_attempt_id=attempt_id,
                    t_actual_ms=10000,
                    correct=True,
                    hint_used=False,
                )
            assert result.q == 5
            assert result.passing is True

        async with session_factory() as session, session.begin():
            failed = await record_card_review(
                session,
                student_id=alice_id,
                question_id=qids[4],
                quiz_attempt_id=attempt_id,
                t_actual_ms=99999,
                correct=False,
                hint_used=False,
            )
            for event in failed.pending_events:
                await dispatcher(
                    session,
                    student_id=event.student_id,
                    question_id=event.question_id,
                    quiz_attempt_id=event.quiz_attempt_id,
                )

        assert failed.q == 0
        assert failed.passing is False
        assert failed.retry_available_at is not None
        assert dispatcher.await_count == 1
        called_kwargs = dispatcher.await_args_list[0].kwargs
        assert called_kwargs["question_id"] == qids[4]
        assert called_kwargs["student_id"] == alice_id

    async with session_factory() as session:
        n_reviews = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM card_reviews "
                    "WHERE student_id = :s AND quiz_attempt_id = :a"
                ),
                {"s": alice_id, "a": attempt_id},
            )
        ).scalar_one()
        n_states = (
            await session.execute(
                text("SELECT COUNT(*) FROM student_card_state WHERE student_id = :s"),
                {"s": alice_id},
            )
        ).scalar_one()
    assert n_reviews == 5
    assert n_states == 5

    async with session_factory() as session:
        unlock = await check_lesson_unlock(
            session, student_id=alice_id, lesson_id=current_lesson_id
        )
    assert unlock.eligible is False
    assert unlock.total_cards == 5
    assert any(b.question_id == qids[4] for b in unlock.blocking_cards)

    async with session_factory() as session:
        next_unlock = await check_lesson_unlock(
            session, student_id=alice_id, lesson_id=next_lesson_id
        )
    assert next_unlock.eligible is False
    assert next_unlock.prereq_lesson_ids_unlocked is False

    advanced_now = datetime.now(tz=UTC) + timedelta(hours=24, minutes=1)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return advanced_now if tz is None else advanced_now.astimezone(tz)  # type: ignore[arg-type]

    with patch(
        "abridgeai.features.spaced_repetition.services.review.datetime",
        _FrozenDatetime,
    ):
        for _ in range(8):
            async with session_factory() as session, session.begin():
                retry = await record_card_review(
                    session,
                    student_id=alice_id,
                    question_id=qids[4],
                    quiz_attempt_id=attempt_id,
                    t_actual_ms=10000,
                    correct=True,
                    hint_used=False,
                )
            if retry.ef_after >= 2.0:
                break
        assert retry.ef_after >= 2.0

    async with session_factory() as session:
        unlock_after = await check_lesson_unlock(
            session, student_id=alice_id, lesson_id=current_lesson_id
        )
    assert unlock_after.eligible is True
    assert unlock_after.passing_cards == 5

    async with session_factory() as session:
        next_unlock_after = await check_lesson_unlock(
            session, student_id=alice_id, lesson_id=next_lesson_id
        )
    assert next_unlock_after.prereq_lesson_ids_unlocked is True

    async with session_factory() as session:
        summary = await student_lesson_summary(
            session, student_id=alice_id, lesson_id=current_lesson_id
        )
    assert summary.kr_estimate > 0.0
    assert summary.progression_ready is True
    assert summary.cards_total == 5

    assert course_id is not None


@pytest.mark.asyncio
async def test_class_wide_difficult_card_detection(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    thesis_seed: dict[str, object],
) -> None:
    """5 students all fail the same card → it surfaces as the top difficult card.

    Also asserts that regenerating the question (simulated by creating a
    replacement ``quiz_questions`` row + retiring the old via
    ``deleted_at``) preserves the original ``CardReview`` audit history.
    """
    course_id = UUID(str(thesis_seed["course_id"]))
    current_lesson_id = UUID(str(thesis_seed["current_lesson_id"]))
    quiz_id = UUID(str(thesis_seed["quiz_id"]))
    qids = [UUID(str(q)) for q in thesis_seed["qids"]]  # type: ignore[union-attr]
    target_qid = qids[0]

    other_students: list[UUID] = []
    async with engine.begin() as conn:
        for _ in range(5):
            sid = uuid.uuid4()
            other_students.append(sid)
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": sid, "email": f"cohort-{sid.hex[:8]}@test.local"},
            )
            await conn.execute(
                text(
                    "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                    "VALUES (:id, :c, :s, 'active')"
                ),
                {"id": uuid.uuid4(), "c": course_id, "s": sid},
            )

    try:
        for sid in other_students:
            attempt_id = await _seed_quiz_attempt(engine, quiz_id=quiz_id, student_id=sid)
            with patch(
                "abridgeai.features.spaced_repetition.services.remediation."
                "dispatch_remediation_for_card_failure",
                new=AsyncMock(),
            ):
                async with session_factory() as session, session.begin():
                    await record_card_review(
                        session,
                        student_id=sid,
                        question_id=target_qid,
                        quiz_attempt_id=attempt_id,
                        t_actual_ms=99999,
                        correct=False,
                        hint_used=False,
                    )

        async with session_factory() as session:
            difficulty = await class_card_difficulty(session, lesson_id=current_lesson_id, top_n=3)
        assert difficulty, "expected at least one difficult card after 5 failures"
        top = difficulty[0]
        assert top.question_id == target_qid
        assert top.mean_ef < 2.0
        assert top.student_count == 5

        replacement_qid = uuid.uuid4()
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions ("
                    "id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs, review_status) "
                    "VALUES ("
                    ":id, :q, 99, 'multiple_choice', 'Q-rescue', "
                    "30000, '[]'::jsonb, 'approved')"
                ),
                {"id": replacement_qid, "q": quiz_id},
            )
            await conn.execute(
                text("UPDATE quiz_questions SET deleted_at = NOW() WHERE id = :q"),
                {"q": target_qid},
            )

        async with engine.begin() as conn:
            history_count = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM card_reviews WHERE question_id = :q"),
                    {"q": target_qid},
                )
            ).scalar_one()
        assert history_count == 5

    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM card_reviews WHERE student_id = ANY(:ids)"),
                {"ids": other_students},
            )
            await conn.execute(
                text("DELETE FROM student_card_state WHERE student_id = ANY(:ids)"),
                {"ids": other_students},
            )
            await conn.execute(
                text("DELETE FROM quiz_attempts WHERE student_id = ANY(:ids)"),
                {"ids": other_students},
            )
            await conn.execute(
                text("DELETE FROM course_enrollments WHERE student_id = ANY(:ids)"),
                {"ids": other_students},
            )
            await conn.execute(
                text("DELETE FROM users WHERE id = ANY(:ids)"),
                {"ids": other_students},
            )


@pytest.mark.asyncio
async def test_compliance_within_grace_window(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    thesis_seed: dict[str, object],
) -> None:
    """Alice reviews 8/10 due cards within grace → ρ=0.8 → not at-risk."""
    alice_id = UUID(str(thesis_seed["alice_id"]))
    course_id = UUID(str(thesis_seed["course_id"]))
    current_lesson_id = UUID(str(thesis_seed["current_lesson_id"]))
    quiz_id = UUID(str(thesis_seed["quiz_id"]))
    seed_qids = [UUID(str(q)) for q in thesis_seed["qids"]]  # type: ignore[union-attr]

    extra_qids = [uuid.uuid4() for _ in range(5)]
    async with engine.begin() as conn:
        for idx, qid in enumerate(extra_qids, start=10):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions ("
                    "id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs, review_status) "
                    "VALUES ("
                    ":id, :quiz, :pos, 'multiple_choice', 'Q?', "
                    "30000, '[]'::jsonb, 'approved')"
                ),
                {"id": qid, "quiz": quiz_id, "pos": idx},
            )

    qids = seed_qids + extra_qids
    due_at = datetime.now(tz=UTC) - timedelta(hours=2)
    async with engine.begin() as conn:
        for qid in qids:
            await conn.execute(
                text(
                    "INSERT INTO student_card_state "
                    "(student_id, question_id, ef, interval_days, repetition_count, "
                    "due_at, total_reviews) "
                    "VALUES (:s, :q, :ef, 1, 1, :due, 1)"
                ),
                {
                    "s": alice_id,
                    "q": qid,
                    "ef": Decimal("2.5"),
                    "due": due_at,
                },
            )
        for qid in qids[:8]:
            await conn.execute(
                text(
                    "INSERT INTO card_reviews ("
                    "id, student_id, question_id, created_at, "
                    "t_actual_ms, t_exp_ms, rho, correct, hint_used, q_derived, "
                    "ef_before, ef_after, interval_before, interval_after, "
                    "n_before, n_after) "
                    "VALUES ("
                    ":id, :s, :q, :ts, "
                    "20000, 30000, 0.6667, TRUE, FALSE, 4, "
                    "2.5, 2.5, 1, 1, 0, 1)"
                ),
                {
                    "id": uuid.uuid4(),
                    "s": alice_id,
                    "q": qid,
                    "ts": due_at + timedelta(hours=1),
                },
            )

    try:
        async with session_factory() as session:
            compliance = await review_compliance_rate(
                session, user_id=alice_id, lesson_id=current_lesson_id
            )
        assert compliance is not None
        assert compliance == pytest.approx(0.8, abs=1e-6)

        async with session_factory() as session:
            at_risk = await at_risk_students(session, course_id=course_id)
        flagged = next((row for row in at_risk if row.student_id == alice_id), None)
        if flagged is not None:
            assert flagged.low_compliance is False
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM quiz_questions WHERE id = ANY(:ids)"),
                {"ids": extra_qids},
            )


__all__: list[str] = []
