"""Integration tests for the teacher-curated knowledge-graph service.

Covers the locked-decision invariants for the CRUD + publish feature:

* first-open seeding returns a valid one-primary draft even with no AI KG
  (``exists=False, seeded=True``);
* :class:`CuratedKGDraftSave` enforces exactly-one-primary / unique ids /
  valid edges at the schema boundary;
* :func:`save_draft` upserts and mirrors ``primary_node_id``;
* :func:`publish` snapshots draft → published, and the learner read
  (:func:`get_published_kg_for_learner`) sees ONLY the published snapshot,
  so unpublished draft edits never leak to students.

Mirrors the raw-SQL setup/teardown pattern of
``test_courses_authoring_service.py`` (avoids cross-feature ORM flush
collisions) and consumes the service layer directly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

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

import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401  -- registers interview_configs FK target
from abridgeai.core.config import get_settings
from abridgeai.features.materials.schemas.curated_kg import (
    CuratedKGDraftSave,
    CuratedKGEdge,
    CuratedKGNode,
)
from abridgeai.features.materials.services import catalog as catalog_service
from abridgeai.features.materials.services.authoring import (
    CuratedKGEmptyError,
    get_or_seed_draft,
    publish,
    save_draft,
    unpublish,
)
from pydantic import ValidationError


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
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict]:
    org_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    lesson_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"kg-{suffix}", "name": "KG Test Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": owner_id, "email": f"kg-{suffix}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'KG Test Course', 'draft')"
            ),
            {"id": course_id, "org": org_id, "owner": owner_id, "slug": f"course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module 1', 1, 'draft')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, lesson_type, status) "
                "VALUES (:id, :module, :slug, 'Lesson 1', 'reading', 'draft')"
            ),
            {"id": lesson_id, "module": module_id, "slug": f"lesson-{suffix}"},
        )

    yield {"owner_id": owner_id, "org_id": org_id, "lesson_id": lesson_id}

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM lesson_knowledge_graphs WHERE lesson_id = :l"),
            {"l": lesson_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :l"), {"l": lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE course_id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


def _draft(*, primary_count: int = 1) -> CuratedKGDraftSave:
    nodes = [
        CuratedKGNode(id="root", label="Root", weight=10, is_primary=primary_count >= 1),
        CuratedKGNode(id="child", label="Child", weight=3, is_primary=primary_count >= 2),
    ]
    return CuratedKGDraftSave(
        nodes=nodes,
        edges=[CuratedKGEdge(source="root", target="child", relation="PREREQUISITE_OF")],
    )


def test_save_rejects_zero_and_multi_primary() -> None:
    with pytest.raises(ValidationError):
        _draft(primary_count=0)
    with pytest.raises(ValidationError):
        _draft(primary_count=2)
    # Exactly one is accepted.
    ok = _draft(primary_count=1)
    assert ok.primary_node_id == "root"


async def test_seed_save_publish_learner_flow(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    lesson_id = scenario["lesson_id"]
    owner_id = scenario["owner_id"]

    # 1) First open: no row yet → seeded placeholder draft with one primary.
    async with session_factory() as session:
        draft = await get_or_seed_draft(session, lesson_id)
    assert draft.exists is False
    assert draft.seeded is True
    assert draft.primary_node_id is not None
    assert sum(1 for n in draft.nodes if n.is_primary) == 1

    # 2) Save a real draft.
    async with session_factory() as session:
        saved = await save_draft(session, lesson_id, _draft(), actor_id=owner_id)
        await session.commit()
    assert saved.exists is True
    assert saved.primary_node_id == "root"
    assert saved.is_published is False
    assert saved.has_unpublished_changes is True

    # 3) Learner sees nothing yet (never published).
    async with session_factory() as session:
        pub = await catalog_service.get_published_kg_for_learner(session, lesson_id)
    assert pub.published is False
    assert pub.nodes == []

    # 4) Publish → learner now sees the snapshot.
    async with session_factory() as session:
        published = await publish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert published.is_published is True
    assert published.has_unpublished_changes is False

    async with session_factory() as session:
        pub = await catalog_service.get_published_kg_for_learner(session, lesson_id)
    assert pub.published is True
    assert pub.primary_node_id == "root"
    assert {n.id for n in pub.nodes} == {"root", "child"}
    assert len(pub.edges) == 1

    # 5) Edit the draft again (remove the edge) but DON'T publish → learner
    #    still sees the OLD published snapshot (draft edits don't leak).
    async with session_factory() as session:
        edited = CuratedKGDraftSave(
            nodes=[CuratedKGNode(id="root", label="Root", weight=10, is_primary=True)],
            edges=[],
        )
        after = await save_draft(session, lesson_id, edited, actor_id=owner_id)
        await session.commit()
    assert after.has_unpublished_changes is True

    async with session_factory() as session:
        pub = await catalog_service.get_published_kg_for_learner(session, lesson_id)
    # Still the published (2-node, 1-edge) snapshot, not the edited draft.
    assert {n.id for n in pub.nodes} == {"root", "child"}
    assert len(pub.edges) == 1


async def test_publish_empty_raises(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    # Publishing with no saved draft row → CuratedKGEmptyError (router maps 409).
    async with session_factory() as session:
        with pytest.raises(CuratedKGEmptyError):
            await publish(session, scenario["lesson_id"], actor_id=scenario["owner_id"])


async def test_seeded_draft_is_not_publishable_until_saved(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Reproduces the 409 a teacher hit when publishing a fresh lesson.

    ``get_or_seed_draft`` returns a NON-EMPTY graph (seeded from the AI KG, or a
    single placeholder primary when there's no AI KG) while ``exists`` is False,
    because the seed is deliberately not persisted. Publishing reads the
    PERSISTED ``draft_json``, so publish-without-save raises even though the UI
    was showing a perfectly good graph.

    The frontend therefore has to save first — see ``handlePublishCurated`` in
    material-hub.tsx, which is what "Save and publish" does. AND a placeholder
    seed (the single "Main concept" node used when the AI graph is empty) can
    never be published even after saving: it has no real content, so it would
    show students a meaningless one-node graph.
    """
    lesson_id = scenario["lesson_id"]
    owner_id = scenario["owner_id"]

    # 1) First open: a usable graph exists in the response, but nothing is saved.
    async with session_factory() as session:
        seeded = await get_or_seed_draft(session, lesson_id)
    assert seeded.exists is False
    assert seeded.seeded is True
    assert seeded.seeded_placeholder is True  # no AI KG in this test env
    assert len(seeded.nodes) > 0, "seed must yield a publishable-looking graph"

    # 2) Publishing right now fails — this is the exact 409 the teacher saw.
    async with session_factory() as session:
        with pytest.raises(CuratedKGEmptyError):
            await publish(session, lesson_id, actor_id=owner_id)

    # 3) Save the seeded graph. When the seed was the placeholder (this test's
    #    AI graph is empty), publishing must STILL fail — the placeholder is
    #    not real content (the saved row no longer says "seeded", but publish
    #    re-detects the placeholder node shape server-side).
    async with session_factory() as session:
        saved = await save_draft(
            session,
            lesson_id,
            CuratedKGDraftSave(nodes=seeded.nodes, edges=seeded.edges),
            actor_id=owner_id,
        )
        await session.commit()
    assert saved.exists is True

    async with session_factory() as session:
        with pytest.raises(CuratedKGEmptyError):
            await publish(session, lesson_id, actor_id=owner_id)

    # 4) Replace the placeholder with a real graph, then publish succeeds.
    async with session_factory() as session:
        await save_draft(
            session,
            lesson_id,
            _draft(),
            actor_id=owner_id,
        )
        await session.commit()

    async with session_factory() as session:
        published = await publish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert published.is_published is True
    assert published.has_unpublished_changes is False
    assert {n.id for n in published.nodes} == {"root", "child"}

    # 5) Students can now see it.
    async with session_factory() as session:
        pub = await catalog_service.get_published_kg_for_learner(session, lesson_id)
    assert pub.published is True
    assert {n.id for n in pub.nodes} == {"root", "child"}


async def test_unpublish_hides_graph_from_learners(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Unpublish rolls the publish back: students lose the panel, the draft
    stays intact for the teacher to re-publish after fixing it."""
    lesson_id = scenario["lesson_id"]
    owner_id = scenario["owner_id"]

    async with session_factory() as session:
        await save_draft(session, lesson_id, _draft(), actor_id=owner_id)
        await session.commit()

    async with session_factory() as session:
        published = await publish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert published.is_published is True

    async with session_factory() as session:
        pub = await catalog_service.get_published_kg_for_learner(session, lesson_id)
    assert pub.published is True

    # Unpublish → learner sees nothing again, draft survives.
    async with session_factory() as session:
        rolled = await unpublish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert rolled.is_published is False
    assert rolled.published_at is None
    assert {n.id for n in rolled.nodes} == {"root", "child"}  # draft intact

    async with session_factory() as session:
        pub = await catalog_service.get_published_kg_for_learner(session, lesson_id)
    assert pub.published is False
    assert pub.nodes == []

    # Re-publishing the untouched draft works.
    async with session_factory() as session:
        repub = await publish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert repub.is_published is True


async def test_republish_after_edit_needs_no_extra_save(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict,
) -> None:
    """Once a row exists, publish alone is enough — no 409.

    This is the branch where the UI shows a plain "Publish" confirmation rather
    than "Save and publish": the draft is already persisted, so publishing just
    snapshots it.
    """
    lesson_id = scenario["lesson_id"]
    owner_id = scenario["owner_id"]

    async with session_factory() as session:
        await save_draft(
            session,
            lesson_id,
            CuratedKGDraftSave(
                nodes=[CuratedKGNode(id="a", label="A", weight=10, is_primary=True)],
                edges=[],
            ),
            actor_id=owner_id,
        )
        await session.commit()

    async with session_factory() as session:
        first = await publish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert first.has_unpublished_changes is False

    # Publishing again with no intervening edit is a harmless no-op re-snapshot.
    async with session_factory() as session:
        again = await publish(session, lesson_id, actor_id=owner_id)
        await session.commit()
    assert again.is_published is True
    assert again.has_unpublished_changes is False
