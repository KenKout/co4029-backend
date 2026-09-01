"""Integration tests for the syllabus-upload MODES (user request 2026-08-31).

Companion to ``test_course_syllabus_import.py``, which covers the original
"create a course" contract (permissions, failure records, transaction shape).
What is new here is that one upload endpoint now does three different things,
and the interesting behaviour is at the seams between them:

* ``attach`` must NOT touch the course. It is the mode a manager picks when
  the course was authored by hand and only the downloadable document is
  missing or stale — if it quietly rewrote the title from a parsed PDF, the
  mode would be a trap. It also must work on a PUBLISHED course (that is the
  point: a live course can finally get its syllabus) and on a PDF the parser
  cannot read (the file is still a fine download).
* ``override`` must REPLACE the learning-outcome tree, not append to it —
  and the old rows must be tombstoned, not deleted, since quiz questions
  reference them.
* ``override`` must be refused with 409 on a non-draft course. Published
  outcomes are the graded scale; this is the same freeze
  ``_assert_outcomes_editable`` enforces for hand edits, and a mode that
  bypassed it would move the goalposts under enrolled students.
* the slug must survive an override: it is in student-visible URLs.

``put_object_bytes`` is monkeypatched — object storage is not what these
tests are about.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import fitz
import httpx
import pytest
import pytest_asyncio
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.access_control.models  # noqa: F401  -- register FK targets
import abridgeai.features.identity.models  # noqa: F401  -- users / storage_objects FK targets
import abridgeai.features.interviews.models  # noqa: F401  -- module_items FK target
import abridgeai.features.materials.models  # noqa: F401  -- lessons FK target
import abridgeai.features.quizzes.models  # noqa: F401  -- module_items FK target
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.courses.routers.authoring import router as authoring_router

pytestmark = pytest.mark.asyncio

_ENDPOINT = "/api/v1/teacher/courses/import-syllabus"

# A real PDF with no text layer: passes the upload validator, fails the parser.
# `attach` must accept it anyway — it never parses.
_NOT_A_SYLLABUS_PDF = b"%PDF-1.4\n% minimal, no text layer\n"


def _syllabus_pdf(*, title: str, outcomes: list[tuple[str, str]]) -> bytes:
    """A minimal ENGLISH syllabus the parser accepts, rendered to a real PDF.

    English only, and generated rather than committed: the fonts PyMuPDF can
    synthesize have no Vietnamese coverage, so a ``vi`` fixture built this way
    would come back as question marks and would be testing the font. The
    Vietnamese field rules are covered by the pure-text parser unit tests.
    """
    lines = [
        "COURSE SYLLABUS",
        "1. Course information",
        "1.1. General information",
        f"- Course title: {title}",
        "Total",
        "90",
        "2. Course description",
        "Topics covered in this course include: consensus and replication.",
        "4.2. Learning outcomes",
        *[f"L.O.{code} - {body}" for code, body in outcomes],
    ]
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 40), "\n".join(lines), fontsize=10)
    return bytes(doc.tobytes())


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def app(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(authoring_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def manager_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    session_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, :exp)"
            ),
            {
                "id": session_id,
                "uid": seeded_users.manager_id,
                "h": hash_secret(generate_token()),
                "exp": datetime.now(tz=UTC) + timedelta(hours=1),
            },
        )
    yield create_access_token(user_id=seeded_users.manager_id, session_id=session_id)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": session_id})


@pytest_asyncio.fixture(autouse=True)
async def _no_object_storage(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Neutralise the S3 put so these tests do not need a live bucket."""

    async def _fake_put(target: object, data: bytes, *, content_type: str) -> None:
        del target, data, content_type

    monkeypatch.setattr(
        "abridgeai.features.courses.services.syllabus_import.put_object_bytes",
        _fake_put,
    )
    yield


@pytest_asyncio.fixture
async def target_course(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[uuid.UUID]:
    """A DRAFT course in the manager's org, with two hand-authored outcomes.

    Its own course rather than the shared seeded one so an override rewriting
    the title cannot leak into another test's expectations. The outcomes exist
    so the "override replaces, does not append" assertion has something to
    replace.
    """
    course_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, "
                "description, status) VALUES (:id, :org, :owner, :slug, :title, :descr, 'draft')"
            ),
            {
                "id": course_id,
                "org": seeded_users.organization_id,
                "owner": seeded_users.manager_id,
                "slug": f"syllabus-mode-{course_id.hex[:8]}",
                "title": "Hand-authored title",
                "descr": "Hand-authored description",
            },
        )
        for position, body in enumerate(("Original outcome A", "Original outcome B"), start=1):
            await conn.execute(
                text(
                    "INSERT INTO course_learning_outcomes "
                    "(id, course_id, parent_id, position, outcome_text) "
                    "VALUES (:id, :course_id, NULL, :position, :body)"
                ),
                {
                    "id": uuid.uuid4(),
                    "course_id": course_id,
                    "position": position,
                    "body": body,
                },
            )
    yield course_id
    async with engine.begin() as conn:
        # Order matters twice here. ``course_id`` on a SUCCEEDED import row is
        # ``ON DELETE SET NULL``, but ``ck_course_syllabus_imports_course_on_success``
        # forbids a succeeded row with a NULL course — so deleting a course
        # BEFORE its import rows makes the cascade violate the constraint.
        # Collect the courses these tests spawned, drop the import rows, then
        # the outcomes, then the courses.
        spawned = [
            row.course_id
            for row in (
                await conn.execute(
                    text(
                        "SELECT DISTINCT course_id FROM course_syllabus_imports "
                        "WHERE course_id IS NOT NULL"
                    )
                )
            ).all()
        ]
        course_ids = {course_id, *spawned}
        await conn.execute(text("DELETE FROM course_syllabus_imports"))
        await conn.execute(
            text("DELETE FROM course_learning_outcomes WHERE course_id = ANY(:ids)"),
            {"ids": list(course_ids)},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": list(course_ids)},
        )


async def _upload(
    client: httpx.AsyncClient,
    bearer: str,
    body: bytes,
    *,
    mode: str,
    course_id: uuid.UUID | None = None,
    language: str = "en",
) -> httpx.Response:
    params = f"?language={language}&filename=syllabus.pdf&mode={mode}"
    if course_id is not None:
        params += f"&course_id={course_id}"
    return await client.post(
        f"{_ENDPOINT}{params}",
        content=body,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/pdf",
        },
    )


async def _course_row(engine: AsyncEngine, course_id: uuid.UUID):
    async with engine.begin() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT title, description, slug, status, estimated_minutes "
                    "FROM courses WHERE id = :id"
                ),
                {"id": course_id},
            )
        ).one()


async def _live_outcomes(engine: AsyncEngine, course_id: uuid.UUID) -> list[str]:
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT outcome_text FROM course_learning_outcomes "
                    "WHERE course_id = :id AND deleted_at IS NULL "
                    "ORDER BY parent_id NULLS FIRST, position"
                ),
                {"id": course_id},
            )
        ).all()
    return [r.outcome_text for r in rows]


async def test_attach_stores_the_document_and_changes_nothing_else(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    target_course: uuid.UUID,
) -> None:
    """`attach` is the "don't touch my course" mode — the whole reason it exists.

    A syllabus that WOULD parse into a different title is used on purpose: if
    attach silently applied it, this test is what catches it.
    """
    pdf = _syllabus_pdf(title="Parsed Replacement Title", outcomes=[("1", "Parsed outcome")])
    response = await _upload(
        client, manager_bearer, pdf, mode="attach", course_id=target_course
    )
    assert response.status_code == 201, response.text

    row = await _course_row(engine, target_course)
    assert row.title == "Hand-authored title"
    assert row.description == "Hand-authored description"
    assert await _live_outcomes(engine, target_course) == [
        "Original outcome A",
        "Original outcome B",
    ]

    # …but the document IS now attached, which is what a student downloads.
    async with engine.begin() as conn:
        stored = (
            await conn.execute(
                text(
                    "SELECT status, storage_object_id FROM course_syllabus_imports "
                    "WHERE course_id = :id ORDER BY created_at DESC LIMIT 1"
                ),
                {"id": target_course},
            )
        ).one()
    assert stored.status == "succeeded"
    assert stored.storage_object_id is not None
    # The response reports the course's EXISTING outcome count, not 0 — an
    # attach did not remove them.
    assert response.json()["outcome_count"] == 2


async def test_attach_accepts_a_pdf_the_parser_cannot_read(
    client: httpx.AsyncClient,
    manager_bearer: str,
    target_course: uuid.UUID,
) -> None:
    """Attach never parses, so an unparseable-but-valid PDF must not 422.

    This is the same file the create path rejects. A manager attaching a
    scanned or oddly-formatted syllabus is publishing a document, not asking
    for fields to be extracted.
    """
    response = await _upload(
        client, manager_bearer, _NOT_A_SYLLABUS_PDF, mode="attach", course_id=target_course
    )
    assert response.status_code == 201, response.text


async def test_override_replaces_the_shell_and_the_outcome_tree(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    target_course: uuid.UUID,
) -> None:
    """Override REPLACES: the old outcomes must be gone, not appended to.

    Appending would double-count the graded scale, and re-numbering would
    collide on the per-parent sibling-position unique index — which is why the
    service tombstones and flushes before inserting.
    """
    pdf = _syllabus_pdf(
        title="Overridden Title",
        outcomes=[("1", "New outcome one"), ("1.1", "New nested outcome"), ("2", "New outcome two")],
    )
    response = await _upload(
        client, manager_bearer, pdf, mode="override", course_id=target_course
    )
    assert response.status_code == 201, response.text

    row = await _course_row(engine, target_course)
    assert row.title == "Overridden Title"
    assert row.estimated_minutes == 90 * 60
    # The slug is student-visible URL surface — an override must not move it.
    assert row.slug.startswith("syllabus-mode-")
    assert row.status == "draft"

    live = await _live_outcomes(engine, target_course)
    assert "Original outcome A" not in live
    assert "Original outcome B" not in live
    assert set(live) == {"New outcome one", "New nested outcome", "New outcome two"}

    # Tombstoned, not hard-deleted: quiz_questions.learning_outcome_id points
    # at these rows and the FK is ON DELETE SET NULL, so a hard delete would
    # silently unmap graded questions.
    async with engine.begin() as conn:
        dead = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM course_learning_outcomes "
                    "WHERE course_id = :id AND deleted_at IS NOT NULL"
                ),
                {"id": target_course},
            )
        ).scalar_one()
    assert dead == 2


async def test_override_on_a_published_course_is_refused_with_409(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    target_course: uuid.UUID,
) -> None:
    """The draft-only rule, enforced server-side and not just dimmed in the UI.

    Published learning outcomes are the graded scale; rewriting them under
    enrolled students is the failure this blocks. 409 (not 422) because
    nothing about the FILE would make the request succeed.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET status = 'published' WHERE id = :id"),
            {"id": target_course},
        )

    pdf = _syllabus_pdf(title="Should Not Land", outcomes=[("1", "Should not land")])
    response = await _upload(
        client, manager_bearer, pdf, mode="override", course_id=target_course
    )
    assert response.status_code == 409, response.text
    assert "course_not_draft" in response.text

    row = await _course_row(engine, target_course)
    assert row.title == "Hand-authored title"
    assert await _live_outcomes(engine, target_course) == [
        "Original outcome A",
        "Original outcome B",
    ]


async def test_attach_is_allowed_on_a_published_course(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    target_course: uuid.UUID,
) -> None:
    """A live course finally getting its syllabus is the headline use case.

    ``attach`` only swaps the downloadable document, so the publish freeze
    does not apply — requiring an unpublish here would take the course away
    from its students to add a PDF.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE courses SET status = 'published' WHERE id = :id"),
            {"id": target_course},
        )

    response = await _upload(
        client, manager_bearer, _NOT_A_SYLLABUS_PDF, mode="attach", course_id=target_course
    )
    assert response.status_code == 201, response.text
    row = await _course_row(engine, target_course)
    assert row.status == "published"


async def test_attach_without_a_course_id_is_rejected(
    client: httpx.AsyncClient, manager_bearer: str
) -> None:
    """No target, no attach — and no attempt row for a request never read."""
    response = await _upload(client, manager_bearer, _NOT_A_SYLLABUS_PDF, mode="attach")
    assert response.status_code == 409
    assert "missing_target_course" in response.text


async def test_unknown_mode_is_rejected_at_the_edge(
    client: httpx.AsyncClient, manager_bearer: str, target_course: uuid.UUID
) -> None:
    """FastAPI validates the Literal, so a typo cannot fall through to a branch."""
    response = await _upload(
        client, manager_bearer, _NOT_A_SYLLABUS_PDF, mode="replace", course_id=target_course
    )
    assert response.status_code == 422


async def test_create_mode_ignores_the_target_course(
    client: httpx.AsyncClient,
    manager_bearer: str,
    engine: AsyncEngine,
    target_course: uuid.UUID,
) -> None:
    """`create` makes a NEW course even when a course_id rides along.

    The dialog offers "Create new course" from a course page, so this
    combination is reachable by design — and must not overwrite the course the
    manager was looking at.
    """
    pdf = _syllabus_pdf(title="Brand New Course", outcomes=[("1", "Fresh outcome")])
    response = await _upload(
        client, manager_bearer, pdf, mode="create", course_id=target_course
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["course_id"] != str(target_course)
    assert body["title"] == "Brand New Course"

    row = await _course_row(engine, target_course)
    assert row.title == "Hand-authored title"
