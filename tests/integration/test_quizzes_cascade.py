from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.courses.models  # noqa: F401  -- register courses / modules FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- ModuleItem -> InterviewConfig relationship target
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz_* FK targets
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
    QuizQuestionRevision,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def org_course(engine: AsyncEngine):
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    suffix = org_id.hex[:8]
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"qz-{suffix}", "name": "Quiz Cascade Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"qz-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, :title, 'draft')"
            ),
            {
                "id": course_id,
                "org": org_id,
                "owner": owner_id,
                "slug": f"qz-course-{suffix}",
                "title": "Quiz Cascade Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, :title, 1, 'draft')"
            ),
            {"id": module_id, "cid": course_id, "title": "Quiz Cascade Module"},
        )
    yield org_id, owner_id, course_id, module_id
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_question_revisions WHERE question_id IN "
                "(SELECT id FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :mid))"
            ),
            {"mid": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE module_id = :mid)"
            ),
            {"mid": module_id},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE module_id = :mid"),
            {"mid": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_soft_delete_cascade_walks_questions_and_revisions(
    session_factory: async_sessionmaker[AsyncSession], org_course
) -> None:
    _org_id, owner_id, course_id, module_id = org_course

    async with session_factory() as session:
        quiz = Quiz(
            course_id=course_id,
            module_id=module_id,
            title="Cascade Test Quiz",
        )
        session.add(quiz)
        await session.flush()

        questions = [
            QuizQuestion(
                quiz_id=quiz.id,
                position=i,
                question_type="multiple_choice",
                prompt_text=f"Question {i}",
            )
            for i in range(1, 3)
        ]
        for question in questions:
            session.add(question)
        await session.flush()

        revisions = [
            QuizQuestionRevision(
                question_id=question.id,
                revision_no=1,
                source_kind="ai",
                payload_json={"prompt": question.prompt_text},
            )
            for question in questions
        ]
        for revision in revisions:
            session.add(revision)
        await session.flush()

        quiz_id = quiz.id
        question_ids = [q.id for q in questions]
        await session.commit()

    async with session_factory() as session:
        quiz = await session.get(Quiz, quiz_id)
        assert quiz is not None
        result = await soft_delete_cascade(session, quiz, actor_id=owner_id)
        await session.commit()

    affected_tables = {tbl for (tbl, _id) in result.affected}
    assert affected_tables == {"quizzes", "quiz_questions"}
    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert quiz_id in affected_ids
    assert all(qid in affected_ids for qid in question_ids)
    assert result.count == 3

    async with session_factory() as session:
        deleted_quiz = (
            await session.execute(
                select(Quiz).where(Quiz.id == quiz_id).execution_options(include_deleted=True)
            )
        ).scalar_one()
        assert deleted_quiz.deleted_at is not None
        assert deleted_quiz.deleted_by == owner_id

        deleted_questions = (
            (
                await session.execute(
                    select(QuizQuestion)
                    .where(QuizQuestion.quiz_id == quiz_id)
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(deleted_questions) == 2
        assert all(q.deleted_at is not None for q in deleted_questions)
        assert all(q.deleted_by == owner_id for q in deleted_questions)

        active_quiz = (
            await session.execute(select(Quiz).where(Quiz.id == quiz_id))
        ).scalar_one_or_none()
        assert active_quiz is None


async def test_delete_question_repacks_sibling_positions(
    session_factory: async_sessionmaker[AsyncSession], org_course
) -> None:
    """Deleting a middle question repacks survivors to a dense 1..N order.

    Regression: the repack UPDATE previously collided with the
    non-deferrable ``uq_quiz_questions_position`` (quiz_id, position)
    unique constraint mid-flush (e.g. shifting 3->2 while a live row still
    held 2), raising a 500. The two-phase offset swap must avoid that.
    """
    from abridgeai.core.security import CurrentUser
    from abridgeai.features.quizzes.services import authoring as authoring_service

    _org_id, owner_id, course_id, module_id = org_course

    async with session_factory() as session:
        quiz = Quiz(course_id=course_id, module_id=module_id, title="Repack Quiz")
        session.add(quiz)
        await session.flush()

        questions = [
            QuizQuestion(
                quiz_id=quiz.id,
                position=i,
                question_type="multiple_choice",
                prompt_text=f"Question {i}",
            )
            for i in range(1, 5)  # positions 1,2,3,4
        ]
        for question in questions:
            session.add(question)
        await session.flush()

        quiz_id = quiz.id
        second_id = questions[1].id  # position 2
        await session.commit()

    actor = CurrentUser(user_id=owner_id, session_id=uuid.uuid4())

    # Delete the middle question — this forces 3->2, 4->3 renumbering.
    async with session_factory() as session:
        await authoring_service.delete_question(session, second_id, actor)
        await session.commit()

    # Survivors must be densely repacked 1..3 with no gaps or collisions.
    async with session_factory() as session:
        survivors = (
            (
                await session.execute(
                    select(QuizQuestion)
                    .where(QuizQuestion.quiz_id == quiz_id)
                    .order_by(QuizQuestion.position)
                )
            )
            .scalars()
            .all()
        )
        assert [q.position for q in survivors] == [1, 2, 3]
        assert second_id not in {q.id for q in survivors}
