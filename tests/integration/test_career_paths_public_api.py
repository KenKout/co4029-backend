"""Coverage for ``career_paths/api/public.py``.

This is the feature's cross-boundary read/write surface: sibling features
(identity's manager/HOD user-detail, learning programs) go through it instead
of reaching into ``queries``/``services``. Two behaviours in it are easy to get
wrong and invisible when they are:

* **Which version a progress read is measured against.** Gap 3 / D3(a) says an
  enrollment stays pinned to the version it started on. So
  ``get_path_course_progress_for_user`` reads the STUDENT'S version when they
  have one and only falls back to the latest published version when they do
  not. Reading the published version for an enrolled student would silently
  re-score them against a route they never walked, every time a manager edits
  the path.

* **The compatibility projection.** Learning-program attempts are the source of
  truth, but the older learner path endpoints still authorize through
  ``student_career_enrollments``. ``ensure_program_path_access`` /
  ``release_program_path_access`` keep that projection in step, and both have
  to be idempotent — they run on every attempt start and end.

Isolation: every test builds its own organization, path, versions and student,
and tears the graph down.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

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

from abridgeai.core.config import get_settings
from abridgeai.features.career_paths.api import public as api


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


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
async def db(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


class Graph:
    """org → student → course → path with v1 published and v2 draft."""

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex[:10]
        self.org_id = uuid.uuid4()
        self.student_id = uuid.uuid4()
        self.actor_id = uuid.uuid4()
        self.course_id = uuid.uuid4()
        self.path_id = uuid.uuid4()
        self.unpublished_path_id = uuid.uuid4()
        self.v1 = uuid.uuid4()  # published
        self.v2 = uuid.uuid4()  # draft
        self.stage_v1 = uuid.uuid4()
        self.stage_v2 = uuid.uuid4()


@pytest_asyncio.fixture
async def graph(engine: AsyncEngine) -> AsyncIterator[Graph]:
    g = Graph()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Career Public API Org', 'active')"
            ),
            {"id": g.org_id, "slug": f"cpapi-{g.tag}"},
        )
        for uid, email in ((g.student_id, "student"), (g.actor_id, "actor")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"cpapi-{g.tag}-{email}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Path Course', 'published')"
            ),
            {
                "id": g.course_id,
                "org": g.org_id,
                "owner": g.actor_id,
                "slug": f"cpapi-course-{g.tag}",
            },
        )
        for pid, slug, name in (
            (g.path_id, f"cpapi-path-{g.tag}", "Published Path"),
            (g.unpublished_path_id, f"cpapi-draft-{g.tag}", "Never Published Path"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                    "VALUES (:id, :org, :slug, :name, 'published')"
                ),
                {"id": pid, "org": g.org_id, "slug": slug, "name": name},
            )
        # v1 published, v2 draft — the pair that makes the pin observable.
        for vid, no, status in ((g.v1, 1, "published"), (g.v2, 2, "draft")):
            await conn.execute(
                # NOTE: :name binds only — this SQLAlchemy's text() no longer
                # recognises %(name)s pyformat placeholders (they pass through
                # literally and psycopg chokes on the "%"). published_at is
                # computed in Python so no bind is reused in two type
                # positions (psycopg3 rejects that as ambiguous).
                text(
                    "INSERT INTO career_path_versions "
                    "(id, career_path_id, version_no, status, published_at) "
                    "VALUES (:id, :path, :no, :status, :published_at)"
                ),
                {
                    "id": vid,
                    "path": g.path_id,
                    "no": no,
                    "status": status,
                    "published_at": datetime.now(tz=UTC) if status == "published" else None,
                },
            )
        for sid, vid in ((g.stage_v1, g.v1), (g.stage_v2, g.v2)):
            await conn.execute(
                text(
                    "INSERT INTO career_path_stages (id, version_id, position, title) "
                    "VALUES (:id, :version, 1, 'Stage one')"
                ),
                {"id": sid, "version": vid},
            )
        # The course sits in v1 only, so a v1 read returns a row and a v2 read
        # returns none — that difference is how the pin is asserted below.
        await conn.execute(
            text(
                "INSERT INTO career_course_items "
                "(version_id, course_id, stage_id, position, is_required) "
                "VALUES (:version, :course, :stage, 1, TRUE)"
            ),
            {"version": g.v1, "course": g.course_id, "stage": g.stage_v1},
        )
    yield g
    async with engine.begin() as conn:
        for stmt in (
            "DELETE FROM student_stage_progress WHERE enrollment_id IN "
            "  (SELECT id FROM student_career_enrollments WHERE student_id = :student)",
            "DELETE FROM student_career_enrollments WHERE student_id = :student",
            "DELETE FROM career_course_items WHERE version_id = ANY(:versions)",
            "DELETE FROM career_path_stages WHERE version_id = ANY(:versions)",
            "DELETE FROM career_path_versions WHERE career_path_id = ANY(:paths)",
            "DELETE FROM career_paths WHERE id = ANY(:paths)",
            "DELETE FROM course_enrollments WHERE course_id = :course",
            "DELETE FROM courses WHERE id = :course",
            "DELETE FROM users WHERE id = ANY(:users)",
            "DELETE FROM organizations WHERE id = :org",
        ):
            await conn.execute(
                text(stmt),
                {
                    "student": g.student_id,
                    "versions": [g.v1, g.v2],
                    "paths": [g.path_id, g.unpublished_path_id],
                    "course": g.course_id,
                    "users": [g.student_id, g.actor_id],
                    "org": g.org_id,
                },
            )


async def _enroll(
    engine: AsyncEngine, graph: Graph, *, version_id: uuid.UUID, status: str = "active"
) -> uuid.UUID:
    enrollment_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO student_career_enrollments "
                "(id, career_path_id, student_id, version_id, status) "
                "VALUES (:id, :path, :student, :version, :status)"
            ),
            {
                "id": enrollment_id,
                "path": graph.path_id,
                "student": graph.student_id,
                "version": version_id,
                "status": status,
            },
        )
    return enrollment_id


# ---------------------------------------------------------------------------
# list_user_career_enrollments
# ---------------------------------------------------------------------------


async def test_lists_the_students_enrollments_with_path_identity(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """The row carries the path's slug and name, not just its id.

    This backs a manager's user-detail panel; returning bare ids would make
    the caller re-query the career_paths feature it was told not to reach
    into.
    """
    await _enroll(engine, graph, version_id=graph.v1)
    rows = await api.list_user_career_enrollments(db, student_id=graph.student_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["career_path_id"] == graph.path_id
    assert row["status"] == "active"
    assert row["name"] == "Published Path"
    assert row["slug"] == f"cpapi-path-{graph.tag}"


async def test_lists_nothing_for_a_student_with_no_enrollments(
    db: AsyncSession, graph: Graph
) -> None:
    del graph
    assert await api.list_user_career_enrollments(db, student_id=uuid.uuid4()) == []


# ---------------------------------------------------------------------------
# get_path_course_progress_for_user — version resolution (D3(a))
# ---------------------------------------------------------------------------


async def test_progress_reads_the_version_the_student_is_pinned_to(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """A student pinned to v2 must be scored against v2, not the published v1.

    The course sits in v1 only, so reading the wrong version is directly
    observable: v1 yields a course row, v2 yields none. If this fell back to
    "latest published" for an enrolled student, every manager edit that
    published a new version would silently re-score everyone mid-path.
    """
    await _enroll(engine, graph, version_id=graph.v2)
    rows = await api.get_path_course_progress_for_user(
        db, career_path_id=graph.path_id, student_id=graph.student_id
    )
    assert rows == [], "v2 has no course items; the pin was ignored"


async def test_progress_uses_the_pinned_version_when_it_is_the_published_one(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    await _enroll(engine, graph, version_id=graph.v1)
    rows = await api.get_path_course_progress_for_user(
        db, career_path_id=graph.path_id, student_id=graph.student_id
    )
    assert [r["course_id"] for r in rows] == [graph.course_id]


async def test_progress_falls_back_to_the_published_version_when_not_enrolled(
    db: AsyncSession, graph: Graph
) -> None:
    """A manager previewing a path for a student who has not started it.

    There is no pin to honour, so the latest published version is the honest
    answer — and it must not be an error.
    """
    rows = await api.get_path_course_progress_for_user(
        db, career_path_id=graph.path_id, student_id=graph.student_id
    )
    assert [r["course_id"] for r in rows] == [graph.course_id]


async def test_progress_is_empty_when_the_path_has_no_published_version(
    db: AsyncSession, graph: Graph
) -> None:
    """No pin and nothing published — return empty rather than raising.

    The caller is rendering a panel; a path still being authored is a normal
    state, not a failure.
    """
    rows = await api.get_path_course_progress_for_user(
        db, career_path_id=graph.unpublished_path_id, student_id=graph.student_id
    )
    assert rows == []


async def test_version_progress_reads_exactly_the_version_it_is_given(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """Program reporting pins its own version and must not infer one.

    Asserted with the student enrolled on the OTHER version, so an
    implementation that quietly consulted the legacy projection would return
    the wrong rows.
    """
    await _enroll(engine, graph, version_id=graph.v2)
    rows = await api.get_version_course_progress_for_user(
        db, version_id=graph.v1, student_id=graph.student_id
    )
    assert [r["course_id"] for r in rows] == [graph.course_id]


# ---------------------------------------------------------------------------
# ensure_program_path_access / release_program_path_access
# ---------------------------------------------------------------------------


async def _enrollment_row(db: AsyncSession, graph: Graph) -> dict | None:
    row = (
        (
            await db.execute(
                text(
                    "SELECT status, version_id, completed_at FROM student_career_enrollments "
                    "WHERE student_id = :s AND career_path_id = :p AND deleted_at IS NULL"
                ),
                {"s": graph.student_id, "p": graph.path_id},
            )
        )
        .mappings()
        .one_or_none()
    )
    return dict(row) if row else None


async def test_ensure_creates_the_projection_when_absent(db: AsyncSession, graph: Graph) -> None:
    await api.ensure_program_path_access(
        db,
        student_id=graph.student_id,
        career_path_id=graph.path_id,
        version_id=graph.v1,
        actor_id=graph.actor_id,
    )
    await db.commit()

    row = await _enrollment_row(db, graph)
    assert row is not None
    assert row["status"] == "active"
    assert row["version_id"] == graph.v1


async def test_ensure_is_idempotent_for_an_already_active_attempt(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """It runs on every attempt start, so a second call must not duplicate.

    ``student_career_enrollments`` is the legacy one-row-per-(student, path)
    projection; a second insert would either violate the constraint or leave
    two rows the learner endpoints cannot choose between.
    """
    await _enroll(engine, graph, version_id=graph.v1)
    await api.ensure_program_path_access(
        db,
        student_id=graph.student_id,
        career_path_id=graph.path_id,
        version_id=graph.v1,
        actor_id=graph.actor_id,
    )
    await db.commit()

    count = (
        await db.execute(
            text(
                "SELECT COUNT(*) FROM student_career_enrollments "
                "WHERE student_id = :s AND career_path_id = :p"
            ),
            {"s": graph.student_id, "p": graph.path_id},
        )
    ).scalar_one()
    assert count == 1


async def test_ensure_reactivates_a_dropped_attempt_onto_the_new_version(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """Re-joining a path is a reactivation, not a second row — and it re-pins.

    A student who dropped and came back is starting the route as it stands
    NOW, so the stale pin and the old completion timestamp both have to go;
    leaving either would score them against a version they abandoned.
    """
    await _enroll(engine, graph, version_id=graph.v1, status="dropped")
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE student_career_enrollments SET completed_at = NOW() "
                "WHERE student_id = :s AND career_path_id = :p"
            ),
            {"s": graph.student_id, "p": graph.path_id},
        )

    await api.ensure_program_path_access(
        db,
        student_id=graph.student_id,
        career_path_id=graph.path_id,
        version_id=graph.v2,
        actor_id=graph.actor_id,
    )
    await db.commit()

    row = await _enrollment_row(db, graph)
    assert row is not None
    assert row["status"] == "active"
    assert row["version_id"] == graph.v2, "reactivation must re-pin to the new version"
    assert row["completed_at"] is None, "a stale completion must be cleared"


async def test_release_drops_an_active_projection(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    await _enroll(engine, graph, version_id=graph.v1)
    await api.release_program_path_access(
        db,
        student_id=graph.student_id,
        career_path_id=graph.path_id,
        actor_id=graph.actor_id,
    )
    await db.commit()

    row = await _enrollment_row(db, graph)
    assert row is not None
    assert row["status"] == "dropped"


async def test_release_leaves_a_completed_attempt_alone(
    db: AsyncSession, engine: AsyncEngine, graph: Graph
) -> None:
    """Only an ACTIVE projection is dropped.

    A finished path is a result worth keeping; rewriting it to "dropped"
    because a program attempt ended would erase the student's completion.
    """
    await _enroll(engine, graph, version_id=graph.v1, status="completed")
    await api.release_program_path_access(
        db,
        student_id=graph.student_id,
        career_path_id=graph.path_id,
        actor_id=graph.actor_id,
    )
    await db.commit()

    row = await _enrollment_row(db, graph)
    assert row is not None
    assert row["status"] == "completed"


async def test_release_is_a_no_op_when_there_is_no_projection(
    db: AsyncSession, graph: Graph
) -> None:
    """It runs at the end of every attempt, including ones that never created
    a projection, so absence must not raise."""
    await api.release_program_path_access(
        db,
        student_id=graph.student_id,
        career_path_id=graph.path_id,
        actor_id=graph.actor_id,
    )
    await db.commit()
    assert await _enrollment_row(db, graph) is None
