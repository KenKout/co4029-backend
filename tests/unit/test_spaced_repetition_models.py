from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import Boolean, DateTime, Integer, Numeric, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.features.spaced_repetition.models import CardReview, StudentCardState


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def quiz_scaffold(engine: AsyncEngine) -> tuple[uuid.UUID, uuid.UUID]:
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {
                "id": org_id,
                "slug": f"sr-{org_id.hex[:8]}",
                "name": "T7.5.1 SR Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"sr-{user_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                "VALUES (:id, :org, :owner, :slug, :title)"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": user_id,
                "slug": f"course-{course_id.hex[:8]}",
                "title": "SR Course",
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
                "INSERT INTO quizzes (id, course_id, module_id, title, status, slug) VALUES (:id, :course, :module, :title, 'draft', 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "id": quiz_id,
                "course": course_id,
                "module": module_id,
                "title": "SR Quiz",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions "
                "(id, quiz_id, position, question_type, prompt_text) "
                "VALUES (:id, :quiz, 1, 'multiple_choice', 'Q?')"
            ),
            {"id": question_id, "quiz": quiz_id},
        )

    yield user_id, question_id

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM card_reviews WHERE student_id = :sid"), {"sid": user_id}
        )
        await conn.execute(
            text("DELETE FROM student_card_state WHERE student_id = :sid OR question_id = :qid"),
            {"sid": user_id, "qid": question_id},
        )
        await conn.execute(text("DELETE FROM quiz_questions WHERE id = :id"), {"id": question_id})
        await conn.execute(text("DELETE FROM quizzes WHERE id = :id"), {"id": quiz_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def test_models_importable() -> None:
    assert StudentCardState.__tablename__ == "student_card_state"
    assert CardReview.__tablename__ == "card_reviews"


def test_student_card_state_composite_pk() -> None:
    pk_cols = {c.name for c in StudentCardState.__table__.primary_key.columns}
    assert pk_cols == {"student_id", "question_id"}, (
        "plan §7.5.1: composite PK on (student_id, question_id), no surrogate id"
    )
    cols = {c.name for c in StudentCardState.__table__.columns}
    assert "id" not in cols, "StudentCardState must not carry a surrogate UUID id"


def test_no_softdelete_on_card_state() -> None:
    cols = {c.name for c in StudentCardState.__table__.columns}
    assert "deleted_at" not in cols, (
        "plan §7.5.1: StudentCardState is overwritten, not soft-deleted"
    )
    assert "deleted_by" not in cols


def test_card_review_immutable() -> None:
    cols = {c.name for c in CardReview.__table__.columns}
    assert "updated_at" not in cols, "plan §7.5.1: CardReview is append-only; CreatedAtMixin only"
    assert "deleted_at" not in cols
    assert "deleted_by" not in cols


def test_card_review_carries_uuid_pk_and_created_at() -> None:
    cols = {c.name: c for c in CardReview.__table__.columns}
    assert "id" in cols
    assert isinstance(cols["created_at"].type, DateTime)
    pk_cols = {c.name for c in CardReview.__table__.primary_key.columns}
    assert pk_cols == {"id"}


def test_student_card_state_column_types() -> None:
    cols = {c.name: c for c in StudentCardState.__table__.columns}
    assert isinstance(cols["ef"].type, Numeric)
    assert cols["ef"].type.precision == 4
    assert cols["ef"].type.scale == 3
    assert isinstance(cols["interval_days"].type, Integer)
    assert isinstance(cols["repetition_count"].type, Integer)
    assert isinstance(cols["due_at"].type, DateTime)
    assert isinstance(cols["calibration_active"].type, Boolean)
    assert isinstance(cols["total_reviews"].type, Integer)


def test_default_ef_is_2_5() -> None:
    col = StudentCardState.__table__.columns["ef"]
    assert col.server_default is not None
    assert "2.5" in str(col.server_default.arg)


def test_default_interval_days_is_1() -> None:
    col = StudentCardState.__table__.columns["interval_days"]
    assert col.server_default is not None
    assert str(col.server_default.arg).strip() == "1"


def test_default_repetition_count_is_0() -> None:
    col = StudentCardState.__table__.columns["repetition_count"]
    assert col.server_default is not None
    assert str(col.server_default.arg).strip() == "0"


def test_default_calibration_active_true() -> None:
    col = StudentCardState.__table__.columns["calibration_active"]
    assert col.server_default is not None
    assert str(col.server_default.arg).strip().upper() == "TRUE"


def test_default_total_reviews_is_0() -> None:
    col = StudentCardState.__table__.columns["total_reviews"]
    assert col.server_default is not None
    assert str(col.server_default.arg).strip() == "0"


def test_ix_student_card_state_due_at_exists() -> None:
    indexes = {idx.name: idx for idx in StudentCardState.__table__.indexes}
    assert "ix_student_card_state_due_at" in indexes
    cols = [c.name for c in indexes["ix_student_card_state_due_at"].columns]
    assert cols == ["student_id", "due_at"], (
        "Index column order matters for the 'due cards' query plan"
    )


def test_ix_card_reviews_student_created_exists() -> None:
    indexes = {idx.name for idx in CardReview.__table__.indexes}
    assert "ix_card_reviews_student_created" in indexes
    assert "ix_card_reviews_question_created" in indexes


async def test_default_ef_is_2_5_on_insert(
    engine: AsyncEngine, quiz_scaffold: tuple[uuid.UUID, uuid.UUID]
) -> None:
    student_id, question_id = quiz_scaffold
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO student_card_state (student_id, question_id) VALUES (:sid, :qid)"),
            {"sid": student_id, "qid": question_id},
        )
        row = (
            await conn.execute(
                text(
                    "SELECT ef, interval_days, repetition_count, calibration_active, "
                    "total_reviews FROM student_card_state "
                    "WHERE student_id = :sid AND question_id = :qid"
                ),
                {"sid": student_id, "qid": question_id},
            )
        ).one()
    assert float(row.ef) == 2.5
    assert row.interval_days == 1
    assert row.repetition_count == 0
    assert row.calibration_active is True
    assert row.total_reviews == 0


async def test_composite_pk_duplicate_raises(
    engine: AsyncEngine, quiz_scaffold: tuple[uuid.UUID, uuid.UUID]
) -> None:
    student_id, question_id = quiz_scaffold
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO student_card_state (student_id, question_id) VALUES (:sid, :qid)"),
            {"sid": student_id, "qid": question_id},
        )
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO student_card_state (student_id, question_id) VALUES (:sid, :qid)"
                ),
                {"sid": student_id, "qid": question_id},
            )


async def test_ef_floor_check_constraint(
    engine: AsyncEngine, quiz_scaffold: tuple[uuid.UUID, uuid.UUID]
) -> None:
    student_id, question_id = quiz_scaffold
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO student_card_state (student_id, question_id, ef) "
                    "VALUES (:sid, :qid, 1.0)"
                ),
                {"sid": student_id, "qid": question_id},
            )


async def test_last_q_check_constraint(
    engine: AsyncEngine, quiz_scaffold: tuple[uuid.UUID, uuid.UUID]
) -> None:
    student_id, question_id = quiz_scaffold
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO student_card_state "
                    "(student_id, question_id, last_q) VALUES (:sid, :qid, 6)"
                ),
                {"sid": student_id, "qid": question_id},
            )


async def test_card_review_q_derived_check_constraint(
    engine: AsyncEngine, quiz_scaffold: tuple[uuid.UUID, uuid.UUID]
) -> None:
    student_id, question_id = quiz_scaffold
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO card_reviews ("
                    "student_id, question_id, t_actual_ms, t_exp_ms, rho, "
                    "correct, hint_used, q_derived, ef_before, ef_after, "
                    "interval_before, interval_after, n_before, n_after"
                    ") VALUES ("
                    ":sid, :qid, 1000, 1000, 1.0, TRUE, FALSE, 6, "
                    "2.5, 2.5, 0, 1, 0, 1)"
                ),
                {"sid": student_id, "qid": question_id},
            )


async def test_fk_cascade_on_student_delete(
    engine: AsyncEngine, quiz_scaffold: tuple[uuid.UUID, uuid.UUID]
) -> None:
    _owner_id, question_id = quiz_scaffold
    student_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": student_id, "email": f"sr-stu-{student_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text("INSERT INTO student_card_state (student_id, question_id) VALUES (:sid, :qid)"),
            {"sid": student_id, "qid": question_id},
        )
        before = (
            await conn.execute(
                text("SELECT count(*) FROM student_card_state WHERE student_id = :sid"),
                {"sid": student_id},
            )
        ).scalar()
    assert before == 1

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM users WHERE id = :sid"), {"sid": student_id})
        after = (
            await conn.execute(
                text("SELECT count(*) FROM student_card_state WHERE student_id = :sid"),
                {"sid": student_id},
            )
        ).scalar()
    assert after == 0, "FK ON DELETE CASCADE must remove the state row when user is removed"
