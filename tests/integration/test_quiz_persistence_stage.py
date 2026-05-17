"""Integration tests for the quiz persistence stage (T5.9).

Covers plan §5817-5828:

* persist_questions creates QuizQuestion + QuizQuestionOption rows
* source_refs are structured with chunk_id metadata
* replace_question_in_place preserves question_id, bumps revision_no
* QuizQuestionRevision row appended on replace
* Stage NEVER commits — caller can rollback after the call
* T0.8 audit listener populates created_by from set_worker_actor
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
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

from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.core.audit import current_actor_var, register_audit_listener
from abridgeai.features.identity import (
    models as _identity_models,  # noqa: F401  -- side-effect: register users table for FK resolution
)
from abridgeai.features.quizzes.ai.stages.persistence import (
    persist_questions,
    replace_question_in_place,
)
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
)
from abridgeai.workers.actor import set_worker_actor

register_audit_listener()


@dataclass
class _Run:
    requested_by: UUID | None


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
    from abridgeai.core.config import get_settings

    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@dataclass(frozen=True)
class _Scaffold:
    org_id: UUID
    user_id: UUID
    course_id: UUID
    module_id: UUID
    quiz: Quiz


@pytest_asyncio.fixture
async def scaffold(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> _Scaffold:
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"t59-{org_id.hex[:8]}", "name": "T5.9 Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"t59-{user_id.hex[:8]}@test.local"},
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
                "title": "T5.9 Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, :pos)"
            ),
            {"id": module_id, "course": course_id, "title": "T5.9 Module", "pos": 1},
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
                "title": "T5.9 Persistence Quiz",
            },
        )

    async with session_factory() as session:
        quiz = (await session.execute(select(Quiz).where(Quiz.id == quiz_id))).scalar_one()
        session.expunge(quiz)

    return _Scaffold(
        org_id=org_id,
        user_id=user_id,
        course_id=course_id,
        module_id=module_id,
        quiz=quiz,
    )


def _chunks() -> tuple[list[ChunkWithDistance], list[str]]:
    chunks = [
        ChunkWithDistance(
            chunk_id=uuid.uuid4(),
            material_version_id=uuid.uuid4(),
            course_id=uuid.uuid4(),
            lesson_id=uuid.uuid4(),
            content=f"chunk-{i}",
            distance=0.1 * i,
        )
        for i in range(3)
    ]
    return chunks, [str(c.chunk_id) for c in chunks]


def _payload(*, chunk_id_refs: list[str], prompt: str = "What is X?") -> dict:
    return {
        "question_type": "multiple_choice",
        "prompt_text": prompt,
        "hint_text": None,
        "explanation": "Because.",
        "difficulty": "medium",
        "bloom_level": "understand",
        "expected_response_time_ms": 30000,
        "source_refs": chunk_id_refs,
        "original_generated_payload": {"raw": "llm-blob"},
        "options": [
            {"option_key": "A", "option_text": "alpha", "is_correct": True, "position": 1},
            {"option_key": "B", "option_text": "beta", "is_correct": False, "position": 2},
        ],
    }


@pytest.mark.asyncio
async def test_persist_creates_question_and_options(
    scaffold: _Scaffold,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    chunks, refs = _chunks()
    payloads = [
        _payload(chunk_id_refs=refs, prompt="Q1?"),
        _payload(chunk_id_refs=refs, prompt="Q2?"),
    ]
    run = _Run(requested_by=scaffold.user_id)

    async with session_factory() as session, session.begin():
        persisted = await persist_questions(session, run, scaffold.quiz, chunks, payloads)
        assert len(persisted) == 2

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(QuizQuestion).where(QuizQuestion.quiz_id == scaffold.quiz.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 2
        prompts = sorted(r.prompt_text for r in rows)
        assert prompts == ["Q1?", "Q2?"]

        opts = (await session.execute(select(QuizQuestionOption))).scalars().all()
        opts_for_quiz = [o for o in opts if o.question_id in {r.id for r in rows}]
        assert len(opts_for_quiz) == 4


@pytest.mark.asyncio
async def test_persist_carries_source_refs(
    scaffold: _Scaffold,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    chunks, refs = _chunks()
    run = _Run(requested_by=scaffold.user_id)

    async with session_factory() as session, session.begin():
        persisted = await persist_questions(
            session,
            run,
            scaffold.quiz,
            chunks,
            [_payload(chunk_id_refs=refs)],
        )
        qid = persisted[0].id

    async with session_factory() as session:
        question = (
            await session.execute(select(QuizQuestion).where(QuizQuestion.id == qid))
        ).scalar_one()
        ref_chunk_ids = {r["chunk_id"] for r in question.source_refs}
        assert ref_chunk_ids == set(refs)
        for ref in question.source_refs:
            assert "material_version_id" in ref


@pytest.mark.asyncio
async def test_replace_in_place_preserves_question_id(
    scaffold: _Scaffold,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    chunks, refs = _chunks()
    run = _Run(requested_by=scaffold.user_id)

    async with session_factory() as session, session.begin():
        persisted = await persist_questions(
            session, run, scaffold.quiz, chunks, [_payload(chunk_id_refs=refs)]
        )
        original_id = persisted[0].id

    async with session_factory() as session, session.begin():
        question = (
            await session.execute(select(QuizQuestion).where(QuizQuestion.id == original_id))
        ).scalar_one()
        replaced = await replace_question_in_place(
            session,
            run,
            question,
            _payload(chunk_id_refs=refs, prompt="REPLACED?"),
            chunks=chunks,
        )
        assert replaced.id == original_id

    async with session_factory() as session:
        revisions = (
            (
                await session.execute(
                    select(QuizQuestionRevision)
                    .where(QuizQuestionRevision.question_id == original_id)
                    .order_by(QuizQuestionRevision.revision_no)
                )
            )
            .scalars()
            .all()
        )
        assert [r.revision_no for r in revisions] == [1, 2]


@pytest.mark.asyncio
async def test_replace_in_place_writes_revision_row(
    scaffold: _Scaffold,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    chunks, refs = _chunks()
    run = _Run(requested_by=scaffold.user_id)

    async with session_factory() as session, session.begin():
        persisted = await persist_questions(
            session, run, scaffold.quiz, chunks, [_payload(chunk_id_refs=refs)]
        )
        qid = persisted[0].id

    async with session_factory() as session, session.begin():
        question = (
            await session.execute(select(QuizQuestion).where(QuizQuestion.id == qid))
        ).scalar_one()
        await replace_question_in_place(
            session,
            run,
            question,
            _payload(chunk_id_refs=refs, prompt="V2?"),
            chunks=chunks,
        )

    async with session_factory() as session:
        rev_count = (
            (
                await session.execute(
                    select(QuizQuestionRevision).where(QuizQuestionRevision.question_id == qid)
                )
            )
            .scalars()
            .all()
        )
        assert len(rev_count) == 2
        assert {r.source_kind for r in rev_count} == {"ai"}


@pytest.mark.asyncio
async def test_persist_does_not_commit(
    scaffold: _Scaffold,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    chunks, refs = _chunks()
    run = _Run(requested_by=scaffold.user_id)

    async with session_factory() as session, session.begin() as txn:
        await persist_questions(session, run, scaffold.quiz, chunks, [_payload(chunk_id_refs=refs)])
        await txn.rollback()

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(QuizQuestion).where(QuizQuestion.quiz_id == scaffold.quiz.id)
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_audit_created_by_via_contextvar(
    scaffold: _Scaffold,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    chunks, refs = _chunks()
    run = _Run(requested_by=scaffold.user_id)

    token = set_worker_actor(scaffold.user_id)
    try:
        async with session_factory() as session, session.begin():
            persisted = await persist_questions(
                session, run, scaffold.quiz, chunks, [_payload(chunk_id_refs=refs)]
            )
            qid = persisted[0].id
    finally:
        current_actor_var.reset(token)

    async with session_factory() as session:
        question = (
            await session.execute(select(QuizQuestion).where(QuizQuestion.id == qid))
        ).scalar_one()
        assert question.created_by == scaffold.user_id
        assert question.updated_by == scaffold.user_id

        options = (
            (
                await session.execute(
                    select(QuizQuestionOption).where(QuizQuestionOption.question_id == qid)
                )
            )
            .scalars()
            .all()
        )
        assert all(opt.created_by == scaffold.user_id for opt in options)

        revision = (
            await session.execute(
                select(QuizQuestionRevision).where(QuizQuestionRevision.question_id == qid)
            )
        ).scalar_one()
        assert revision.created_by == scaffold.user_id
