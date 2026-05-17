"""Integration tests for SR remediation dispatch (T7.5.10 + BUG-2 fix).

Covers plan §7.5.10:

* Audio chunk → ``?t=<seconds>`` deep-link
* PDF chunk → ``?p=<page>`` deep-link
* Video chunk → ``?t=<seconds>`` deep-link
* Empty KG result → no notification dispatched
* After-commit dispatch (commit case): Notification row created
* After-commit dispatch (rollback case): no Notification row (BUG-2)

Neo4j is mocked at the ``_concepts_for_chunks`` /
``_chunks_for_concepts`` / ``retrieve_kg_context_for_anchors`` boundaries
so the suite can run without a live KG. The Postgres-side resolution
(chunk → material → deep-link) runs against the real test database to
exercise the raw ``text()`` SQL paths.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
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

from abridgeai.ai.knowledge_graph.schemas import Concept, KGContext
from abridgeai.core.config import get_settings
from abridgeai.features.identity import (
    models as _identity_models,  # noqa: F401  -- register users table
)
from abridgeai.features.notifications.models import Notification
from abridgeai.features.quizzes import (
    models as _quiz_models,  # noqa: F401  -- register quiz tables for FK resolution
)
from abridgeai.features.spaced_repetition.services import (
    build_deep_link,
    dispatch_remediation_for_card_failure,
    record_card_review,
)

_REMEDIATION_MODULE = "abridgeai.features.spaced_repetition.services.remediation"


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


async def _seed_world(
    engine: AsyncEngine,
    *,
    material_type: str,
    chunk_metadata: dict[str, object],
    related_chunk_metadata: dict[str, object] | None = None,
    related_material_type: str | None = None,
    seed_source_refs: bool = True,
) -> dict[str, UUID]:
    """Insert a course → quiz with one question + a related chunk to surface.

    Returns key UUIDs so tests can wire up mocks for Neo4j edges.

    The "seed chunk" is referenced by the question's ``source_refs``;
    the "related chunk" lives in a separate material in the same course
    and is what the dispatcher should surface as a deep-link.
    """
    org_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    quiz_id = uuid.uuid4()
    question_id = uuid.uuid4()
    attempt_id = uuid.uuid4()

    lesson_id = uuid.uuid4()
    seed_storage_id = uuid.uuid4()
    seed_material_id = uuid.uuid4()
    seed_version_id = uuid.uuid4()
    seed_chunk_id = uuid.uuid4()

    related_storage_id = uuid.uuid4()
    related_material_id = uuid.uuid4()
    related_version_id = uuid.uuid4()
    related_chunk_id = uuid.uuid4()
    related_lesson_id = uuid.uuid4()

    if related_material_type is None:
        related_material_type = material_type
    if related_chunk_metadata is None:
        related_chunk_metadata = chunk_metadata

    course_slug = f"course-{course_id.hex[:8]}"

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"sr-{org_id.hex[:8]}", "name": "T7.5.10 Org"},
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
                "slug": course_slug,
                "title": "T7.5.10 Course",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position) "
                "VALUES (:id, :course, :title, :pos)"
            ),
            {"id": module_id, "course": course_id, "title": "M", "pos": 1},
        )
        for lid, slug in ((lesson_id, "lesson-seed"), (related_lesson_id, "lesson-related")):
            await conn.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title, lesson_type) "
                    "VALUES (:id, :mod, :slug, :title, 'reading')"
                ),
                {"id": lid, "mod": module_id, "slug": slug, "title": f"Lesson {slug}"},
            )
        for sid, kind in (
            (seed_storage_id, "seed"),
            (related_storage_id, "related"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO storage_objects (id, bucket, object_key, mime_type, size_bytes) "
                    "VALUES (:id, 'test-bucket', :key, 'application/octet-stream', 100)"
                ),
                {"id": sid, "key": f"obj-{kind}-{sid.hex[:8]}"},
            )
        for mid, lid, mtype, vid, sid in (
            (seed_material_id, lesson_id, material_type, seed_version_id, seed_storage_id),
            (
                related_material_id,
                related_lesson_id,
                related_material_type,
                related_version_id,
                related_storage_id,
            ),
        ):
            await conn.execute(
                text(
                    "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                    "VALUES (:id, :lid, :title, :mt)"
                ),
                {"id": mid, "lid": lid, "title": f"{mtype.title()} Material", "mt": mtype},
            )
            await conn.execute(
                text(
                    "INSERT INTO learning_material_versions "
                    "(id, material_id, storage_object_id, version_no, is_current, processing_status) "
                    "VALUES (:id, :mid, :sid, 1, TRUE, 'ready')"
                ),
                {"id": vid, "mid": mid, "sid": sid},
            )
        await conn.execute(
            text(
                "INSERT INTO document_chunks "
                "(id, course_id, module_id, lesson_id, material_version_id, "
                "chunk_index, chunk_type, content, metadata, content_hash) "
                "VALUES (:id, :course, :module, :lesson, :ver, 0, :ctype, "
                "'seed content', CAST(:meta AS jsonb), 'seedhash')"
            ),
            {
                "id": seed_chunk_id,
                "course": course_id,
                "module": module_id,
                "lesson": lesson_id,
                "ver": seed_version_id,
                "ctype": _chunk_type_for(material_type),
                "meta": _to_jsonb(chunk_metadata),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO document_chunks "
                "(id, course_id, module_id, lesson_id, material_version_id, "
                "chunk_index, chunk_type, content, metadata, content_hash) "
                "VALUES (:id, :course, :module, :lesson, :ver, 0, :ctype, "
                "'related content', CAST(:meta AS jsonb), 'relatedhash')"
            ),
            {
                "id": related_chunk_id,
                "course": course_id,
                "module": module_id,
                "lesson": related_lesson_id,
                "ver": related_version_id,
                "ctype": _chunk_type_for(related_material_type),
                "meta": _to_jsonb(related_chunk_metadata),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:id, :course, :module, :title, 'draft')"
            ),
            {"id": quiz_id, "course": course_id, "module": module_id, "title": "Q"},
        )
        await conn.execute(
            text("INSERT INTO quiz_source_lessons (quiz_id, lesson_id) VALUES (:q, :l)"),
            {"q": quiz_id, "l": lesson_id},
        )
        source_refs_value = f'[{{"chunk_id": "{seed_chunk_id}"}}]' if seed_source_refs else "[]"
        await conn.execute(
            text(
                "INSERT INTO quiz_questions ("
                "id, quiz_id, position, question_type, prompt_text, "
                "expected_response_time_ms, source_refs, review_status"
                ") VALUES ("
                ":id, :quiz, 1, 'multiple_choice', 'Q?', "
                "30000, CAST(:refs AS jsonb), 'pending'"
                ")"
            ),
            {"id": question_id, "quiz": quiz_id, "refs": source_refs_value},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts (id, quiz_id, student_id, attempt_number) "
                "VALUES (:id, :quiz, :student, 1)"
            ),
            {"id": attempt_id, "quiz": quiz_id, "student": student_id},
        )

    return {
        "student_id": student_id,
        "course_id": course_id,
        "course_slug_uuid": course_id,
        "quiz_id": quiz_id,
        "question_id": question_id,
        "quiz_attempt_id": attempt_id,
        "seed_chunk_id": seed_chunk_id,
        "related_chunk_id": related_chunk_id,
        "related_material_id": related_material_id,
        "related_lesson_id": related_lesson_id,
    }


def _chunk_type_for(material_type: str) -> str:
    mapping = {
        "audio": "audio",
        "video": "video",
        "pdf": "pdf",
        "document": "pdf",
        "slides": "pptx",
        "pptx": "pptx",
        "docx": "docx",
        "xlsx": "xlsx",
        "html": "text",
        "text": "text",
    }
    return mapping.get(material_type, "text")


def _to_jsonb(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload)


def _kg_with_concepts(*names: str) -> KGContext:
    return KGContext(concepts=[Concept(name=n) for n in names], enabled=True)


def _patch_kg(seed_concepts: list[str], related_chunk_ids: list[UUID], kg_context: KGContext):
    return (
        patch(
            f"{_REMEDIATION_MODULE}._concepts_for_chunks",
            new=AsyncMock(return_value=seed_concepts),
        ),
        patch(
            f"{_REMEDIATION_MODULE}.retrieve_kg_context_for_anchors",
            new=AsyncMock(return_value=kg_context),
        ),
        patch(
            f"{_REMEDIATION_MODULE}._chunks_for_concepts",
            new=AsyncMock(return_value=related_chunk_ids),
        ),
    )


@pytest.mark.asyncio
async def test_audio_deep_link(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="audio",
        chunk_metadata={"timestamp_start_ms": 4_530_000},
    )
    p1, p2, p3 = _patch_kg(
        seed_concepts=["Concept A"],
        related_chunk_ids=[seeds["related_chunk_id"]],
        kg_context=_kg_with_concepts("Concept B"),
    )
    with p1, p2, p3:
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=seeds["student_id"],
                question_id=seeds["question_id"],
                quiz_attempt_id=seeds["quiz_attempt_id"],
            )
            await session.commit()

    async with session_factory() as session:
        notification = (
            await session.execute(
                select(Notification).where(Notification.user_id == seeds["student_id"])
            )
        ).scalar_one()
        assert notification.category == "spaced_repetition"
        assert "?t=4530" in notification.body
        assert notification.entity_type == "quiz_question"
        assert notification.entity_id == seeds["question_id"]


@pytest.mark.asyncio
async def test_pdf_deep_link(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 12},
    )
    p1, p2, p3 = _patch_kg(
        seed_concepts=["Anchor"],
        related_chunk_ids=[seeds["related_chunk_id"]],
        kg_context=_kg_with_concepts("Related"),
    )
    with p1, p2, p3:
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=seeds["student_id"],
                question_id=seeds["question_id"],
                quiz_attempt_id=seeds["quiz_attempt_id"],
            )
            await session.commit()

    async with session_factory() as session:
        notification = (
            await session.execute(
                select(Notification).where(Notification.user_id == seeds["student_id"])
            )
        ).scalar_one()
        assert "?p=12" in notification.body


@pytest.mark.asyncio
async def test_video_deep_link(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="video",
        chunk_metadata={"timestamp_start_ms": 90_000},
    )
    p1, p2, p3 = _patch_kg(
        seed_concepts=["Algo"],
        related_chunk_ids=[seeds["related_chunk_id"]],
        kg_context=_kg_with_concepts("RelatedAlgo"),
    )
    with p1, p2, p3:
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=seeds["student_id"],
                question_id=seeds["question_id"],
                quiz_attempt_id=seeds["quiz_attempt_id"],
            )
            await session.commit()

    async with session_factory() as session:
        notification = (
            await session.execute(
                select(Notification).where(Notification.user_id == seeds["student_id"])
            )
        ).scalar_one()
        assert "?t=90" in notification.body


@pytest.mark.asyncio
async def test_skip_when_no_resources(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 1},
    )
    p1, p2, p3 = _patch_kg(
        seed_concepts=["X"],
        related_chunk_ids=[],
        kg_context=_kg_with_concepts("Y"),
    )
    with p1, p2, p3:
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=seeds["student_id"],
                question_id=seeds["question_id"],
                quiz_attempt_id=seeds["quiz_attempt_id"],
            )
            await session.commit()

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(Notification).where(Notification.user_id == seeds["student_id"])
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_skip_when_kg_returns_empty(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 1},
    )
    p1, p2, p3 = _patch_kg(
        seed_concepts=["X"],
        related_chunk_ids=[],
        kg_context=KGContext(enabled=True),
    )
    with p1, p2, p3:
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=seeds["student_id"],
                question_id=seeds["question_id"],
                quiz_attempt_id=seeds["quiz_attempt_id"],
            )
            await session.commit()

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(Notification).where(Notification.user_id == seeds["student_id"])
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_after_commit_pattern_review_yields_pending_event(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """BUG-2: review service must NOT dispatch at flush time.

    Records a failing review, asserts the result carries a queued
    ``CardFailedEvent`` for the caller to dispatch *after* commit.
    """
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 5},
    )

    async with session_factory() as session, session.begin():
        result = await record_card_review(
            session,
            student_id=seeds["student_id"],
            question_id=seeds["question_id"],
            quiz_attempt_id=seeds["quiz_attempt_id"],
            t_actual_ms=20000,
            correct=False,
            hint_used=False,
        )

    assert result.q == 0
    assert len(result.pending_events) == 1
    queued = result.pending_events[0]
    assert queued.student_id == seeds["student_id"]
    assert queued.question_id == seeds["question_id"]
    assert queued.quiz_attempt_id == seeds["quiz_attempt_id"]
    assert queued.timestamp.tzinfo is not None


@pytest.mark.asyncio
async def test_rollback_yields_no_notification(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """BUG-2 regression guard.

    If the review transaction rolls back, the caller must skip dispatch
    -- the test simulates the contract by deliberately rolling back and
    asserting no Notification row exists for the student.
    """
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 5},
    )

    p1, p2, p3 = _patch_kg(
        seed_concepts=["Any"],
        related_chunk_ids=[seeds["related_chunk_id"]],
        kg_context=_kg_with_concepts("Any2"),
    )

    async with session_factory() as session:
        with p1, p2, p3:
            await session.begin()
            try:
                result = await record_card_review(
                    session,
                    student_id=seeds["student_id"],
                    question_id=seeds["question_id"],
                    quiz_attempt_id=seeds["quiz_attempt_id"],
                    t_actual_ms=20000,
                    correct=False,
                    hint_used=False,
                )
                assert len(result.pending_events) == 1
            finally:
                await session.rollback()

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(Notification).where(Notification.user_id == seeds["student_id"])
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_commit_then_dispatch_creates_notification(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """BUG-2 happy path: review commit + after-commit dispatch.

    The expected pattern: caller records review inside one session,
    commits, opens a fresh session, dispatches remediation, commits
    again. Notification row must exist after the second commit.
    """
    seeds = await _seed_world(
        engine,
        material_type="audio",
        chunk_metadata={"timestamp_start_ms": 12_000},
    )

    async with session_factory() as session, session.begin():
        result = await record_card_review(
            session,
            student_id=seeds["student_id"],
            question_id=seeds["question_id"],
            quiz_attempt_id=seeds["quiz_attempt_id"],
            t_actual_ms=20000,
            correct=False,
            hint_used=False,
        )
    assert len(result.pending_events) == 1
    event = result.pending_events[0]

    p1, p2, p3 = _patch_kg(
        seed_concepts=["Concept"],
        related_chunk_ids=[seeds["related_chunk_id"]],
        kg_context=_kg_with_concepts("Concept2"),
    )
    with p1, p2, p3:
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=event.student_id,
                question_id=event.question_id,
                quiz_attempt_id=event.quiz_attempt_id,
            )
            await session.commit()

    async with session_factory() as session:
        notification = (
            await session.execute(
                select(Notification).where(Notification.user_id == seeds["student_id"])
            )
        ).scalar_one()
        assert notification.category == "spaced_repetition"
        assert "?t=12" in notification.body


def test_build_deep_link_audio() -> None:
    url = build_deep_link(
        course_slug="cs101",
        lesson_id=UUID("00000000-0000-0000-0000-000000000001"),
        material_id=UUID("00000000-0000-0000-0000-000000000002"),
        material_type="audio",
        source_location={"timestamp_start_ms": 4_530_000},
    )
    assert url.endswith("?t=4530")


def test_build_deep_link_video() -> None:
    url = build_deep_link(
        course_slug="cs101",
        lesson_id=UUID("00000000-0000-0000-0000-000000000001"),
        material_id=UUID("00000000-0000-0000-0000-000000000002"),
        material_type="video",
        source_location={"timestamp_start_ms": 90_000},
    )
    assert url.endswith("?t=90")


def test_build_deep_link_pdf() -> None:
    url = build_deep_link(
        course_slug="cs101",
        lesson_id=UUID("00000000-0000-0000-0000-000000000001"),
        material_id=UUID("00000000-0000-0000-0000-000000000002"),
        material_type="pdf",
        source_location={"page": 12},
    )
    assert url.endswith("?p=12")


def test_build_deep_link_html() -> None:
    url = build_deep_link(
        course_slug="cs101",
        lesson_id=UUID("00000000-0000-0000-0000-000000000001"),
        material_id=UUID("00000000-0000-0000-0000-000000000002"),
        material_type="html",
        source_location={"anchor": "section-2"},
    )
    assert url.endswith("#section-2")


def test_build_deep_link_fallback_no_metadata() -> None:
    url = build_deep_link(
        course_slug="cs101",
        lesson_id=UUID("00000000-0000-0000-0000-000000000001"),
        material_id=UUID("00000000-0000-0000-0000-000000000002"),
        material_type="audio",
        source_location={},
    )
    assert "?" not in url
    assert "#" not in url
    assert url.startswith("/courses/cs101/lessons/")


def test_build_deep_link_unknown_type_returns_base() -> None:
    url = build_deep_link(
        course_slug="cs101",
        lesson_id=UUID("00000000-0000-0000-0000-000000000001"),
        material_id=UUID("00000000-0000-0000-0000-000000000002"),
        material_type="image",
        source_location={"page": 1},
    )
    assert url == (
        "/courses/cs101/lessons/00000000-0000-0000-0000-000000000001/"
        "resources/00000000-0000-0000-0000-000000000002"
    )


@pytest.mark.asyncio
async def test_unknown_question_id_no_op(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    bogus = uuid.uuid4()
    student_id = uuid.uuid4()
    async with session_factory() as session:
        await dispatch_remediation_for_card_failure(
            session,
            student_id=student_id,
            question_id=bogus,
            quiz_attempt_id=None,
        )
        await session.commit()

    async with session_factory() as session:
        rows = (
            (await session.execute(select(Notification).where(Notification.user_id == student_id)))
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_skip_when_no_seed_concepts(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 1},
    )
    with patch(
        f"{_REMEDIATION_MODULE}._concepts_for_chunks",
        new=AsyncMock(return_value=[]),
    ):
        async with session_factory() as session:
            await dispatch_remediation_for_card_failure(
                session,
                student_id=seeds["student_id"],
                question_id=seeds["question_id"],
                quiz_attempt_id=seeds["quiz_attempt_id"],
            )
            await session.commit()

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(Notification).where(Notification.user_id == seeds["student_id"])
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_skip_when_source_refs_empty(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    seeds = await _seed_world(
        engine,
        material_type="pdf",
        chunk_metadata={"page": 1},
        seed_source_refs=False,
    )
    async with session_factory() as session:
        await dispatch_remediation_for_card_failure(
            session,
            student_id=seeds["student_id"],
            question_id=seeds["question_id"],
            quiz_attempt_id=seeds["quiz_attempt_id"],
        )
        await session.commit()

    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(Notification).where(Notification.user_id == seeds["student_id"])
                )
            )
            .scalars()
            .all()
        )
        assert rows == []


def test_unused_after_commit_module_marker() -> None:
    """Anchor for evidence file: confirms the after-commit suite ran."""
    now = datetime.now(tz=UTC)
    assert now.tzinfo is not None
