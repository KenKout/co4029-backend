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

import abridgeai.ai.models  # noqa: F401  -- register generation_runs FK target
import abridgeai.features.courses.models  # noqa: F401  -- register courses / modules FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- register interview_* FK targets
import abridgeai.features.quizzes.models  # noqa: F401  -- register quiz_attempts FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
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
            {"id": org_id, "slug": f"int-{suffix}", "name": "Interview Cascade Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"int-{suffix}@test.local"},
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
                "slug": f"int-course-{suffix}",
                "title": "Interview Cascade Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :cid, :title, 1, 'draft')"
            ),
            {"id": module_id, "cid": course_id, "title": "Interview Cascade Module"},
        )
    yield org_id, owner_id, course_id, module_id
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM interview_outcome_evaluations WHERE outcome_id IN "
                "(SELECT id FROM interview_outcomes WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :mid))"
            ),
            {"mid": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_questions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :mid)"
            ),
            {"mid": module_id},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcomes WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE module_id = :mid)"
            ),
            {"mid": module_id},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE module_id = :mid"),
            {"mid": module_id},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def test_soft_delete_cascade_walks_questions_and_outcomes(
    session_factory: async_sessionmaker[AsyncSession], org_course
) -> None:
    org_id, owner_id, course_id, module_id = org_course

    async with session_factory() as session:
        config = InterviewConfig(
            course_id=course_id,
            module_id=module_id,
            slug="cascade-test-config",
            title="Cascade Test Config",
        )
        session.add(config)
        await session.flush()

        outcomes = [
            InterviewOutcome(
                interview_config_id=config.id,
                position=i,
                outcome_text=f"Outcome {i}",
                outcome_type="knowledge",
            )
            for i in range(1, 3)
        ]
        for outcome in outcomes:
            session.add(outcome)
        questions = [
            InterviewQuestion(
                interview_config_id=config.id,
                position=i,
                question_type="conceptual",
                prompt_text=f"Question {i}",
            )
            for i in range(1, 3)
        ]
        for question in questions:
            session.add(question)
        await session.flush()

        config_id = config.id
        question_ids = [q.id for q in questions]
        outcome_ids = [o.id for o in outcomes]
        await session.commit()

    async with session_factory() as session:
        config = await session.get(InterviewConfig, config_id)
        assert config is not None
        result = await soft_delete_cascade(session, config, actor_id=owner_id)
        await session.commit()

    affected_tables = {tbl for (tbl, _id) in result.affected}
    assert affected_tables == {
        "interview_configs",
        "interview_questions",
        "interview_outcomes",
    }
    affected_ids = {id_ for (_tbl, id_) in result.affected}
    assert config_id in affected_ids
    assert all(qid in affected_ids for qid in question_ids)
    assert all(oid in affected_ids for oid in outcome_ids)
    assert result.count == 5

    async with session_factory() as session:
        deleted_config = (
            await session.execute(
                select(InterviewConfig)
                .where(InterviewConfig.id == config_id)
                .execution_options(include_deleted=True)
            )
        ).scalar_one()
        assert deleted_config.deleted_at is not None
        assert deleted_config.deleted_by == owner_id

        deleted_questions = (
            (
                await session.execute(
                    select(InterviewQuestion)
                    .where(InterviewQuestion.interview_config_id == config_id)
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(deleted_questions) == 2
        assert all(q.deleted_at is not None for q in deleted_questions)
        assert all(q.deleted_by == owner_id for q in deleted_questions)

        deleted_outcomes = (
            (
                await session.execute(
                    select(InterviewOutcome)
                    .where(InterviewOutcome.interview_config_id == config_id)
                    .execution_options(include_deleted=True)
                )
            )
            .scalars()
            .all()
        )
        assert len(deleted_outcomes) == 2
        assert all(o.deleted_at is not None for o in deleted_outcomes)
        assert all(o.deleted_by == owner_id for o in deleted_outcomes)

        active_config = (
            await session.execute(select(InterviewConfig).where(InterviewConfig.id == config_id))
        ).scalar_one_or_none()
        assert active_config is None
