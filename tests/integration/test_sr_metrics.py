"""Integration tests for SR performance metric queries (T7.5.7).

Covers plan §7.5.7:
* KR estimate: empty lesson → 0.0; uniform EF → expected scaled value.
* Progression readiness: delegates to T7.5.6 ``check_lesson_unlock``.
* Review compliance: no due cards → None; full window → 1.0; partial → correct fraction.
* Composite ``student_lesson_summary`` glues the three together.
* Class-wide histogram bucketing across lesson cards.
* ``class_card_difficulty`` returns top-N lowest mean EF.
* ``at_risk_students`` flags low-compliance signal.

Each test seeds its own organization → users → course → module → lesson →
quiz → questions tree, so cache keys (``kr:<user>:<lesson>``,
``compliance:<user>:<lesson>``) are unique per test and never collide.
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
from abridgeai.features.identity import models as _identity_models  # noqa: F401
from abridgeai.features.quizzes import models as _quiz_models  # noqa: F401
from abridgeai.features.spaced_repetition.queries import (
    StudentLessonSummary,
    at_risk_students,
    class_card_difficulty,
    class_kr_distribution,
    knowledge_retention_estimate,
    progression_readiness,
    review_compliance_rate,
    student_lesson_summary,
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
    """Force cache miss so tests are deterministic without flushing Redis."""
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


async def _seed_lesson(
    engine: AsyncEngine,
    *,
    n_questions: int = 0,
    course_id: UUID | None = None,
    organization_id: UUID | None = None,
    student_id: UUID | None = None,
    lesson_status: str = "published",
    ef_min_unlock: Decimal = Decimal("2.0"),
    tau_unlock: Decimal = Decimal("0.8"),
    requires_interview_pass: bool = False,
) -> tuple[UUID, UUID, UUID, list[UUID]]:
    """Seed a complete tree and return (student_id, course_id, lesson_id, [question_ids])."""
    org_id = organization_id or uuid.uuid4()
    sid = student_id or uuid.uuid4()
    cid = course_id or uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_ids = [uuid.uuid4() for _ in range(n_questions)]

    async with engine.begin() as conn:
        if organization_id is None:
            await conn.execute(
                text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
                {"id": org_id, "slug": f"sr-met-{org_id.hex[:8]}", "name": "SR T7.5.7"},
            )
        if student_id is None:
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": sid, "email": f"sr-met-{sid.hex[:8]}@test.local"},
            )
        if course_id is None:
            await conn.execute(
                text(
                    "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                    "VALUES (:id, :org, :owner, :slug, :title)"
                ),
                {
                    "id": cid,
                    "org": org_id,
                    "owner": sid,
                    "slug": f"course-{cid.hex[:8]}",
                    "title": "T7.5.7 Course",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO course_enrollments "
                    "(id, course_id, student_id, status) "
                    "VALUES (:id, :course, :student, 'active')"
                ),
                {"id": uuid.uuid4(), "course": cid, "student": sid},
            )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, :title, :pos, 'published')"
            ),
            {"id": module_id, "course": cid, "title": "M", "pos": 1},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status, ef_min_unlock, tau_unlock, "
                "requires_interview_pass) "
                "VALUES (:id, :m, :slug, :title, :status, :ef_min, :tau, :iv)"
            ),
            {
                "id": lesson_id,
                "m": module_id,
                "slug": f"lesson-{lesson_id.hex[:8]}",
                "title": "L",
                "status": lesson_status,
                "ef_min": ef_min_unlock,
                "tau": tau_unlock,
                "iv": requires_interview_pass,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :m, :title, 'published')"
            ),
            {"id": quiz_id, "course": cid, "m": module_id, "title": "Q"},
        )
        await conn.execute(
            text("INSERT INTO quiz_source_lessons (quiz_id, lesson_id) VALUES (:q, :l)"),
            {"q": quiz_id, "l": lesson_id},
        )
        for idx, q in enumerate(question_ids, start=1):
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions ("
                    "id, quiz_id, position, question_type, prompt_text, "
                    "expected_response_time_ms, source_refs, review_status"
                    ") VALUES ("
                    ":id, :quiz, :pos, 'multiple_choice', 'Q?', "
                    "30000, '[]'::jsonb, 'approved'"
                    ")"
                ),
                {"id": q, "quiz": quiz_id, "pos": idx},
            )
    return sid, cid, lesson_id, question_ids


async def _set_card_state(
    engine: AsyncEngine,
    *,
    student_id: UUID,
    question_id: UUID,
    ef: Decimal,
    due_at: datetime | None = None,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO student_card_state ("
                "student_id, question_id, ef, interval_days, repetition_count, "
                "due_at, total_reviews"
                ") VALUES ("
                ":sid, :qid, :ef, 1, 1, COALESCE(:due, NOW()), 1"
                ") ON CONFLICT (student_id, question_id) DO UPDATE "
                "SET ef = EXCLUDED.ef, due_at = EXCLUDED.due_at"
            ),
            {"sid": student_id, "qid": question_id, "ef": ef, "due": due_at},
        )


async def _add_card_review(
    engine: AsyncEngine,
    *,
    student_id: UUID,
    question_id: UUID,
    created_at: datetime,
    correct: bool = True,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO card_reviews ("
                "id, student_id, question_id, created_at, "
                "t_actual_ms, t_exp_ms, rho, correct, hint_used, q_derived, "
                "ef_before, ef_after, interval_before, interval_after, "
                "n_before, n_after"
                ") VALUES ("
                ":id, :sid, :qid, :ts, "
                "20000, 30000, 0.6667, :ok, FALSE, :q, "
                "2.5, 2.5, 1, 1, 0, 1"
                ")"
            ),
            {
                "id": uuid.uuid4(),
                "sid": student_id,
                "qid": question_id,
                "ts": created_at,
                "ok": correct,
                "q": 4 if correct else 0,
            },
        )


@pytest.mark.asyncio
async def test_kr_empty_lesson_returns_zero(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _course_id, lesson_id, _ = await _seed_lesson(engine, n_questions=0)
    async with session_factory() as session:
        kr = await knowledge_retention_estimate(session, user_id=student_id, lesson_id=lesson_id)
    assert kr == 0.0


@pytest.mark.asyncio
async def test_kr_all_cards_at_initial_ef_returns_one(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=5)
    for qid in qids:
        await _set_card_state(engine, student_id=student_id, question_id=qid, ef=Decimal("2.5"))
    async with session_factory() as session:
        kr = await knowledge_retention_estimate(session, user_id=student_id, lesson_id=lesson_id)
    assert kr == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_kr_all_cards_at_min_ef_returns_zero(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=5)
    for qid in qids:
        await _set_card_state(engine, student_id=student_id, question_id=qid, ef=Decimal("1.3"))
    async with session_factory() as session:
        kr = await knowledge_retention_estimate(session, user_id=student_id, lesson_id=lesson_id)
    assert kr == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_kr_in_unit_range(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=4)
    for qid, ef in zip(
        qids, [Decimal("1.3"), Decimal("1.9"), Decimal("2.2"), Decimal("2.5")], strict=True
    ):
        await _set_card_state(engine, student_id=student_id, question_id=qid, ef=ef)
    async with session_factory() as session:
        kr = await knowledge_retention_estimate(session, user_id=student_id, lesson_id=lesson_id)
    assert 0.0 <= kr <= 1.0
    assert kr == pytest.approx(((0.0) + (0.5) + (0.75) + (1.0)) / 4, abs=1e-6)


@pytest.mark.asyncio
async def test_progression_readiness_delegates_to_unlock_gate(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, _ = await _seed_lesson(engine, n_questions=0)
    async with session_factory() as session:
        ready = await progression_readiness(session, student_id=student_id, lesson_id=lesson_id)
    assert ready is True


@pytest.mark.asyncio
async def test_compliance_no_due_cards_returns_none(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, _ = await _seed_lesson(engine, n_questions=3)
    async with session_factory() as session:
        compliance = await review_compliance_rate(session, user_id=student_id, lesson_id=lesson_id)
    assert compliance is None


@pytest.mark.asyncio
async def test_compliance_full_window_returns_one(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=3)
    due_in_past = datetime.now(tz=UTC) - timedelta(hours=2)
    review_at = due_in_past + timedelta(hours=1)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.0"),
            due_at=due_in_past,
        )
        await _add_card_review(engine, student_id=student_id, question_id=qid, created_at=review_at)
    async with session_factory() as session:
        compliance = await review_compliance_rate(session, user_id=student_id, lesson_id=lesson_id)
    assert compliance == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_compliance_partial_returns_correct_fraction(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=4)
    due_in_past = datetime.now(tz=UTC) - timedelta(hours=2)
    review_at = due_in_past + timedelta(hours=1)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.0"),
            due_at=due_in_past,
        )
    for qid in qids[:1]:
        await _add_card_review(engine, student_id=student_id, question_id=qid, created_at=review_at)
    async with session_factory() as session:
        compliance = await review_compliance_rate(session, user_id=student_id, lesson_id=lesson_id)
    assert compliance == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_student_lesson_summary_composes_all_three(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=2)
    for qid in qids:
        await _set_card_state(engine, student_id=student_id, question_id=qid, ef=Decimal("2.5"))
    async with session_factory() as session:
        summary = await student_lesson_summary(session, student_id=student_id, lesson_id=lesson_id)
    assert isinstance(summary, StudentLessonSummary)
    assert summary.kr_estimate == pytest.approx(1.0)
    assert summary.progression_ready is True
    assert summary.cards_total == 2


@pytest.mark.asyncio
async def test_class_kr_distribution_histogram_buckets(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    org_id = uuid.uuid4()
    course_id = uuid.uuid4()
    s1 = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"sr-cls-{org_id.hex[:8]}", "name": "SR T7.5.7 cls"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :e)"),
            {"id": s1, "e": f"s1-{s1.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :o, :u, :slug, 'C')"
            ),
            {"id": course_id, "o": org_id, "u": s1, "slug": f"c-{course_id.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                "VALUES (:id, :c, :s, 'active')"
            ),
            {"id": uuid.uuid4(), "c": course_id, "s": s1},
        )
    _, _, lesson_id, qids = await _seed_lesson(
        engine,
        n_questions=2,
        course_id=course_id,
        organization_id=org_id,
        student_id=s1,
    )
    s2 = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :e)"),
            {"id": s2, "e": f"s2-{s2.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status) "
                "VALUES (:id, :c, :s, 'active')"
            ),
            {"id": uuid.uuid4(), "c": course_id, "s": s2},
        )
    for qid in qids:
        await _set_card_state(engine, student_id=s1, question_id=qid, ef=Decimal("2.5"))
        await _set_card_state(engine, student_id=s2, question_id=qid, ef=Decimal("1.3"))
    async with session_factory() as session:
        dist = await class_kr_distribution(session, course_id=course_id, lesson_id=lesson_id)
    assert dist.student_count == 2
    assert len(dist.histogram) == 10
    bucket_counts = {round(lo, 2): cnt for lo, cnt in dist.histogram}
    assert bucket_counts[0.0] == 1
    assert bucket_counts[0.9] == 1
    assert dist.mean_kr == pytest.approx(0.5, abs=1e-6)


@pytest.mark.asyncio
async def test_class_card_difficulty_returns_top_n_lowest(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, _cid, lesson_id, qids = await _seed_lesson(engine, n_questions=4)
    for qid, ef in zip(
        qids, [Decimal("1.4"), Decimal("2.5"), Decimal("1.3"), Decimal("2.0")], strict=True
    ):
        await _set_card_state(engine, student_id=student_id, question_id=qid, ef=ef)
    async with session_factory() as session:
        ranked = await class_card_difficulty(session, lesson_id=lesson_id, top_n=2)
    assert len(ranked) == 2
    assert ranked[0].question_id == qids[2]
    assert ranked[1].question_id == qids[0]
    assert ranked[0].mean_ef < ranked[1].mean_ef


@pytest.mark.asyncio
async def test_at_risk_low_compliance_signal(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    student_id, course_id, _lesson_id, qids = await _seed_lesson(engine, n_questions=4)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO user_profiles "
                "(user_id, given_name, family_name, display_name) "
                "VALUES (:u, 'At', 'Risk', 'At Risk Student') "
                "ON CONFLICT (user_id) DO NOTHING"
            ),
            {"u": student_id},
        )
    due_in_past = datetime.now(tz=UTC) - timedelta(days=3)
    for qid in qids:
        await _set_card_state(
            engine,
            student_id=student_id,
            question_id=qid,
            ef=Decimal("2.0"),
            due_at=due_in_past,
        )
    async with session_factory() as session:
        flagged = await at_risk_students(session, course_id=course_id)
    matches = [s for s in flagged if s.student_id == student_id]
    assert len(matches) == 1
    risk = matches[0]
    assert risk.low_compliance is True
    assert risk.frozen_kr is True
    assert risk.high_theory_practice_gap is False
