"""Deleting a quiz must not leave a ghost item in the course content tree.

Reported symptom: ``GET /teacher/courses/{id}/content`` listed a quiz whose
detail endpoint returned 404. The UI rendered the entry, and clicking it failed.

Two independent defects, both fixed:

1. ``quizzes.services.authoring.delete_quiz`` soft-deleted the quiz (and
   cascaded to questions/options/revisions) but left the ``module_items`` row
   that points at it alive. ``soft_delete_cascade`` only walks ONETOMANY
   relationships and ``module_items -> quizzes`` is MANYTOONE from the item
   side (``Quiz`` has no ``items`` relationship), so the cascade could never
   reach it. ``delete_lesson`` / ``delete_module`` already handled this;
   ``delete_quiz`` was the sole omission.

2. ``courses.services.authoring.get_authoring_content`` correctly skipped the
   soft-deleted quiz when resolving ``target``, but then appended the item
   ANYWAY with ``target=None`` and a dangling ``quiz_id``. Now such items are
   skipped. A DB CHECK guarantees every ``item_type`` carries exactly one
   non-null target reference, so this can only ever drop genuinely dangling
   rows, never legitimate content.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import Column, Table, select, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- FK targets
import abridgeai.features.courses.models  # noqa: F401  -- FK targets
import abridgeai.features.identity.models  # noqa: F401  -- users FK target
import abridgeai.features.interviews.models  # noqa: F401  -- interview_* tables
from abridgeai.core.config import get_settings
from abridgeai.core.db import Base
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.models import ModuleItem
from abridgeai.features.courses.services import authoring as courses_authoring
from abridgeai.features.quizzes.services import authoring as quiz_authoring

for _stub in ("interview_configs", "learning_materials", "learning_material_versions"):
    if _stub not in Base.metadata.tables:
        Table(_stub, Base.metadata, Column("id", PGUUID(as_uuid=True), primary_key=True))


def _async_url(url: str) -> str:
    if "+psycopg_async" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return url


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
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
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id, owner_id = uuid.uuid4(), uuid.uuid4()
    course_id, module_id = uuid.uuid4(), uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:i,:s,'Ghost Org')"),
            {"i": org_id, "s": f"gh-{suffix}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:i,:e)"),
            {"i": owner_id, "e": f"gh-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status)"
                " VALUES (:i,:o,:u,:s,'Ghost Course','draft')"
            ),
            {"i": course_id, "o": org_id, "u": owner_id, "s": f"course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status)"
                " VALUES (:i,:c,'Module',1,'draft')"
            ),
            {"i": module_id, "c": course_id},
        )

    yield {"course_id": course_id, "module_id": module_id, "owner_id": owner_id}

    async with engine.begin() as conn:
        for stmt in (
            "DELETE FROM quiz_question_revisions WHERE question_id IN (SELECT id FROM quiz_questions WHERE quiz_id IN (SELECT id FROM quizzes WHERE module_id=:m))",
            "DELETE FROM quiz_questions WHERE quiz_id IN (SELECT id FROM quizzes WHERE module_id=:m)",
            "DELETE FROM module_items WHERE module_id=:m",
            "DELETE FROM quizzes WHERE module_id=:m",
            "DELETE FROM modules WHERE id=:m",
        ):
            await conn.execute(text(stmt), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id=:i"), {"i": course_id})
        await conn.execute(text("DELETE FROM users WHERE id=:i"), {"i": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id=:i"), {"i": org_id})


def _actor(user_id: uuid.UUID) -> CurrentUser:
    return CurrentUser(user_id=user_id, session_id=uuid.uuid4())


class _QuizPayload:
    def __init__(self, **f: object) -> None:
        self._f = f

    def model_dump(self, exclude_unset: bool = False) -> dict:
        del exclude_unset
        return dict(self._f)


def _quiz_ids_in_tree(tree: dict) -> list[uuid.UUID]:
    out = []
    for module in tree.get("modules") or []:
        for item in module.get("items") or []:
            if item.get("quiz_id"):
                out.append(item["quiz_id"])
    return out


@pytest.mark.asyncio
async def test_deleted_quiz_disappears_from_content_tree(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict
) -> None:
    actor = _actor(scenario["owner_id"])

    async with session_factory() as db:
        quiz = await quiz_authoring.create_quiz(
            db,
            scenario["module_id"],
            _QuizPayload(
                course_id=scenario["course_id"],
                module_id=scenario["module_id"],
                title="Ghost Quiz",
            ),
            actor,
        )
        await db.commit()
        quiz_id = quiz.id

    # Present before deletion.
    async with session_factory() as db:
        tree = await courses_authoring.get_authoring_content(db, scenario["course_id"])
    assert quiz_id in _quiz_ids_in_tree(tree)

    async with session_factory() as db:
        await quiz_authoring.delete_quiz(db, quiz_id, actor)
        await db.commit()

    # Defect 1: the module_item must be soft-deleted too.
    async with session_factory() as db:
        item_deleted_at = (
            await db.execute(
                select(ModuleItem.deleted_at).where(ModuleItem.quiz_id == quiz_id)
            )
        ).scalars().all()
    assert all(d is not None for d in item_deleted_at), (
        "module_items row still alive after the quiz was deleted — the content "
        "tree will keep serving a dangling quiz_id"
    )

    # Defect 2: and it must not appear in the tree at all.
    async with session_factory() as db:
        tree_after = await courses_authoring.get_authoring_content(
            db, scenario["course_id"]
        )
    assert quiz_id not in _quiz_ids_in_tree(tree_after)

    # No item may ever be emitted without a resolvable target.
    for module in tree_after.get("modules") or []:
        for item in module.get("items") or []:
            assert item.get("target") is not None, (
                f"item {item.get('id')} emitted with target=None — the client "
                "renders it and the detail endpoint 404s"
            )


@pytest.mark.asyncio
async def test_soft_deleted_quiz_row_is_not_reachable(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict
) -> None:
    """The quiz itself stays soft-deleted (tombstoned, not hard-deleted)."""
    actor = _actor(scenario["owner_id"])
    async with session_factory() as db:
        quiz = await quiz_authoring.create_quiz(
            db,
            scenario["module_id"],
            _QuizPayload(
                course_id=scenario["course_id"],
                module_id=scenario["module_id"],
                title="Tombstone Quiz",
            ),
            actor,
        )
        await db.commit()
        quiz_id = quiz.id

    async with session_factory() as db:
        await quiz_authoring.delete_quiz(db, quiz_id, actor)
        await db.commit()

    # RAW SQL on purpose: a global ``do_orm_execute`` soft-delete filter
    # (core.db.soft_delete) rewrites every ORM query to exclude tombstoned
    # rows, so an ORM select can never observe ``deleted_at`` being set.
    async with session_factory() as db:
        deleted_at = (
            await db.execute(
                text("SELECT deleted_at FROM quizzes WHERE id = :i"), {"i": quiz_id}
            )
        ).scalar()
    assert deleted_at is not None, "quiz should be tombstoned, not hard-deleted"


@pytest.mark.asyncio
async def test_orphaned_module_item_is_not_emitted(
    session_factory: async_sessionmaker[AsyncSession], scenario: dict
) -> None:
    """Defence in depth for pre-existing orphans.

    Rows created before the ``delete_quiz`` cascade fix (or by any other writer
    that tombstones a quiz without its item) must still not surface. Simulates
    that state directly: quiz tombstoned, module_item left alive.
    """
    actor = _actor(scenario["owner_id"])
    async with session_factory() as db:
        quiz = await quiz_authoring.create_quiz(
            db,
            scenario["module_id"],
            _QuizPayload(
                course_id=scenario["course_id"],
                module_id=scenario["module_id"],
                title="Orphan Quiz",
            ),
            actor,
        )
        await db.commit()
        quiz_id = quiz.id

    # Tombstone ONLY the quiz, leaving the module_item alive (the old bug's
    # end state).
    async with session_factory() as db:
        await db.execute(
            text("UPDATE quizzes SET deleted_at = now() WHERE id = :i"), {"i": quiz_id}
        )
        await db.commit()

    async with session_factory() as db:
        item_alive = (
            await db.execute(
                text(
                    "SELECT count(*) FROM module_items "
                    "WHERE quiz_id = :i AND deleted_at IS NULL"
                ),
                {"i": quiz_id},
            )
        ).scalar()
    assert item_alive == 1, "precondition: the orphaned item must still be live"

    async with session_factory() as db:
        tree = await courses_authoring.get_authoring_content(db, scenario["course_id"])

    assert quiz_id not in _quiz_ids_in_tree(tree), (
        "an orphaned module_item leaked into the content tree — the client will "
        "render it and 404 on click"
    )
