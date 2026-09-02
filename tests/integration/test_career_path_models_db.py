"""Structural invariants of ``features/career_paths/models.py``.

The docstrings in that module make a set of load-bearing claims about what the
SCHEMA makes impossible -- not what a service happens to check. Those two
drift: a service guard can be refactored away, a migration can quietly relax a
constraint, and the model file keeps asserting a guarantee that no longer
exists. Each test here pins one of those claims to the database itself.

The claims, in the module's own terms:

* A version pins a route. ``(career_path_id, version_no)`` is unique and
  ``status`` is a two-value enum, because an enrollment pins to a version and
  a duplicate or unknown-status version makes that pin ambiguous.
* ``career_course_items`` is keyed on ``(version_id, course_id)`` -- described
  as "what makes 'the same course in two stages of one VERSION' structurally
  impossible", with the same course across two versions explicitly allowed.
* ``student_stage_progress`` is an append-only latch that deliberately omits
  ``SoftDeleteMixin``, because "adding SoftDeleteMixin would let a
  soft-deleted latch row un-complete a student".
* Readiness scores are a percentage and are constrained to 0..100.

Isolation: every test builds its own org/course/path graph and tears it down.
Constraint violations are asserted by name so a test cannot pass on the wrong
error -- a NOT NULL slip would otherwise read as a satisfied CHECK.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.features.career_paths.models import (
    CareerPathCourse,
    CareerPathStage,
    CareerPathVersion,
    CareerReadinessSnapshot,
    StudentStageProgress,
)


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


class Graph:
    """org -> course(s) -> career_path -> version -> stage, plus a student."""

    def __init__(self) -> None:
        self.org_id = uuid.uuid4()
        self.student_id = uuid.uuid4()
        self.course_a = uuid.uuid4()
        self.course_b = uuid.uuid4()
        self.path_id = uuid.uuid4()
        self.version_id = uuid.uuid4()
        self.version_two_id = uuid.uuid4()
        self.stage_id = uuid.uuid4()
        self.stage_two_id = uuid.uuid4()
        self.enrollment_id = uuid.uuid4()


@pytest_asyncio.fixture
async def graph(engine: AsyncEngine) -> AsyncIterator[Graph]:
    g = Graph()
    tag = g.org_id.hex[:10]
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Career Model Org', 'active')"
            ),
            {"id": g.org_id, "slug": f"career-model-{tag}"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": g.student_id, "email": f"career-{tag}@test.local"},
        )
        for course_id, suffix in ((g.course_a, "a"), (g.course_b, "b")):
            await conn.execute(
                text(
                    "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                    "VALUES (:id, :org, :owner, :slug, 'Career Model Course')"
                ),
                {
                    "id": course_id,
                    "org": g.org_id,
                    "owner": g.student_id,
                    "slug": f"career-course-{tag}-{suffix}",
                },
            )
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, 'Career Model Path', 'draft')"
            ),
            {"id": g.path_id, "org": g.org_id, "slug": f"career-path-{tag}"},
        )
        for version_id, version_no in ((g.version_id, 1), (g.version_two_id, 2)):
            await conn.execute(
                text(
                    "INSERT INTO career_path_versions "
                    "(id, career_path_id, version_no, status) "
                    "VALUES (:id, :path, :no, 'draft')"
                ),
                {"id": version_id, "path": g.path_id, "no": version_no},
            )
        for stage_id, version_id in (
            (g.stage_id, g.version_id),
            (g.stage_two_id, g.version_id),
        ):
            await conn.execute(
                text(
                    "INSERT INTO career_path_stages (id, version_id, position, title) "
                    "VALUES (:id, :version, :pos, 'S')"
                ),
                {
                    "id": stage_id,
                    "version": version_id,
                    "pos": 1 if stage_id == g.stage_id else 2,
                },
            )
        await conn.execute(
            text(
                "INSERT INTO student_career_enrollments "
                "(id, career_path_id, student_id, version_id, status) "
                "VALUES (:id, :path, :student, :version, 'active')"
            ),
            {
                "id": g.enrollment_id,
                "path": g.path_id,
                "student": g.student_id,
                "version": g.version_id,
            },
        )
    yield g
    async with engine.begin() as conn:
        for stmt in (
            "DELETE FROM student_stage_progress WHERE enrollment_id = :enrollment",
            "DELETE FROM career_readiness_snapshots WHERE student_id = :student",
            "DELETE FROM student_career_enrollments WHERE career_path_id = :path",
            "DELETE FROM career_course_items WHERE version_id = ANY(:versions)",
            "DELETE FROM career_path_stages WHERE version_id = ANY(:versions)",
            "DELETE FROM career_path_versions WHERE career_path_id = :path",
            "DELETE FROM career_paths WHERE id = :path",
            "DELETE FROM courses WHERE id = ANY(:courses)",
            "DELETE FROM users WHERE id = :student",
            "DELETE FROM organizations WHERE id = :org",
        ):
            await conn.execute(
                text(stmt),
                {
                    "enrollment": g.enrollment_id,
                    "student": g.student_id,
                    "path": g.path_id,
                    "versions": [g.version_id, g.version_two_id],
                    "courses": [g.course_a, g.course_b],
                    "org": g.org_id,
                },
            )


async def _expect_violation(engine: AsyncEngine, constraint: str, sql: str, params: dict) -> None:
    """Run ``sql`` and require it to fail on ``constraint`` specifically.

    Naming the constraint matters: without it a statement that failed for an
    unrelated reason (a typo, a NOT NULL, a missing FK) reads as the CHECK
    doing its job, and the test passes while proving nothing.
    """
    with pytest.raises(IntegrityError) as excinfo:
        async with engine.begin() as conn:
            await conn.execute(text(sql), params)
    assert constraint in str(excinfo.value), f"expected {constraint}, got: {excinfo.value}"


# ---------------------------------------------------------------------------
# career_path_versions -- the pin an enrollment holds
# ---------------------------------------------------------------------------


async def test_a_version_number_is_unique_within_its_path(
    engine: AsyncEngine, graph: Graph
) -> None:
    """Two rows claiming to be "v1 of this path" make an enrollment's pin ambiguous."""
    await _expect_violation(
        engine,
        "career_path_versions_path_version_key",
        "INSERT INTO career_path_versions (id, career_path_id, version_no, status) "
        "VALUES (gen_random_uuid(), :path, 1, 'draft')",
        {"path": graph.path_id},
    )


async def test_the_same_version_number_is_free_on_another_path(
    engine: AsyncEngine, graph: Graph
) -> None:
    """Uniqueness is per path, not global -- every path starts at v1."""
    other_path = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:id, :org, :slug, 'Second Path', 'draft')"
            ),
            {
                "id": other_path,
                "org": graph.org_id,
                "slug": f"career-path-2-{other_path.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO career_path_versions (id, career_path_id, version_no, status) "
                "VALUES (gen_random_uuid(), :path, 1, 'draft')"
            ),
            {"path": other_path},
        )
        await conn.execute(
            text("DELETE FROM career_path_versions WHERE career_path_id = :p"),
            {"p": other_path},
        )
        await conn.execute(text("DELETE FROM career_paths WHERE id = :p"), {"p": other_path})


@pytest.mark.parametrize("status", ["archived", "published ", "PUBLISHED", ""])
async def test_version_status_is_a_closed_two_value_enum(
    engine: AsyncEngine, graph: Graph, status: str
) -> None:
    """Only ``published`` may be pinned to, so a third state has no meaning.

    The whitespace and casing cases are the realistic ones: they come from
    hand-written migrations and data fixes, not from the application.
    """
    await _expect_violation(
        engine,
        "ck_career_path_versions_status",
        "INSERT INTO career_path_versions (id, career_path_id, version_no, status) "
        "VALUES (gen_random_uuid(), :path, 99, :status)",
        {"path": graph.path_id, "status": status},
    )


@pytest.mark.parametrize("version_no", [0, -1])
async def test_version_numbers_are_positive(
    engine: AsyncEngine, graph: Graph, version_no: int
) -> None:
    await _expect_violation(
        engine,
        "career_path_versions_version_no_check",
        "INSERT INTO career_path_versions (id, career_path_id, version_no, status) "
        "VALUES (gen_random_uuid(), :path, :no, 'draft')",
        {"path": graph.path_id, "no": version_no},
    )


# ---------------------------------------------------------------------------
# career_path_stages -- the gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, -1])
async def test_stage_positions_are_positive(
    engine: AsyncEngine, graph: Graph, position: int
) -> None:
    await _expect_violation(
        engine,
        "career_path_stages_position_check",
        "INSERT INTO career_path_stages (id, version_id, position, title) "
        "VALUES (gen_random_uuid(), :version, :pos, 'S')",
        {"version": graph.version_two_id, "pos": position},
    )


async def test_min_optional_to_complete_cannot_be_negative(
    engine: AsyncEngine, graph: Graph
) -> None:
    """A negative requirement would make a stage complete before it began."""
    await _expect_violation(
        engine,
        "career_path_stages_min_optional_to_complete_check",
        "INSERT INTO career_path_stages "
        "(id, version_id, position, title, min_optional_to_complete) "
        "VALUES (gen_random_uuid(), :version, 5, 'S', -1)",
        {"version": graph.version_two_id},
    )


@pytest.mark.parametrize("policy", ["always", "after_previous", "after_previous_required"])
async def test_every_documented_unlock_policy_is_accepted(
    engine: AsyncEngine, graph: Graph, policy: str
) -> None:
    """The allowed set is asserted from both sides.

    Checking only the rejections would let the CHECK be narrowed to a single
    value without any test noticing -- and the gating code would then start
    failing on policies the model file still documents.
    """
    stage_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, version_id, position, title, unlock_policy) "
                "VALUES (:id, :version, 7, 'S', :policy)"
            ),
            {"id": stage_id, "version": graph.version_two_id, "policy": policy},
        )
        await conn.execute(text("DELETE FROM career_path_stages WHERE id = :id"), {"id": stage_id})


async def test_an_unknown_unlock_policy_is_rejected(engine: AsyncEngine, graph: Graph) -> None:
    await _expect_violation(
        engine,
        "career_path_stages_unlock_policy_check",
        "INSERT INTO career_path_stages (id, version_id, position, title, unlock_policy) "
        "VALUES (gen_random_uuid(), :version, 8, 'S', 'after_next')",
        {"version": graph.version_two_id},
    )


@pytest.mark.parametrize("enforcement", ["hard", "soft", "advisory"])
async def test_every_documented_enforcement_level_is_accepted(
    engine: AsyncEngine, graph: Graph, enforcement: str
) -> None:
    stage_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, version_id, position, title, enforcement) "
                "VALUES (:id, :version, 9, 'S', :enforcement)"
            ),
            {"id": stage_id, "version": graph.version_two_id, "enforcement": enforcement},
        )
        await conn.execute(text("DELETE FROM career_path_stages WHERE id = :id"), {"id": stage_id})


async def test_an_unknown_enforcement_level_is_rejected(engine: AsyncEngine, graph: Graph) -> None:
    """A stage that is neither blocking nor advisory has no defined behaviour."""
    await _expect_violation(
        engine,
        "career_path_stages_enforcement_check",
        "INSERT INTO career_path_stages (id, version_id, position, title, enforcement) "
        "VALUES (gen_random_uuid(), :version, 10, 'S', 'blocking')",
        {"version": graph.version_two_id},
    )


# ---------------------------------------------------------------------------
# career_course_items -- the claim the module comments lean on hardest
# ---------------------------------------------------------------------------


async def _add_item(
    engine: AsyncEngine,
    *,
    version_id: uuid.UUID,
    course_id: uuid.UUID,
    stage_id: uuid.UUID,
    position: int,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_course_items "
                "(version_id, course_id, stage_id, position) "
                "VALUES (:version, :course, :stage, :position)"
            ),
            {
                "version": version_id,
                "course": course_id,
                "stage": stage_id,
                "position": position,
            },
        )


async def test_one_course_cannot_appear_twice_in_one_version(
    engine: AsyncEngine, graph: Graph
) -> None:
    """The composite primary key is doing real work here.

    The module comment says re-keying on ``(stage_id, course_id)`` would
    permit this. It is the same course placed in two DIFFERENT stages of one
    version -- which would make a student's progress in it count twice toward
    the path.
    """
    await _add_item(
        engine,
        version_id=graph.version_id,
        course_id=graph.course_a,
        stage_id=graph.stage_id,
        position=1,
    )
    await _expect_violation(
        engine,
        "career_course_items_pkey",
        "INSERT INTO career_course_items (version_id, course_id, stage_id, position) "
        "VALUES (:version, :course, :stage, 1)",
        {
            "version": graph.version_id,
            "course": graph.course_a,
            "stage": graph.stage_two_id,
        },
    )


async def test_the_same_course_may_appear_in_two_versions_of_one_path(
    engine: AsyncEngine, graph: Graph
) -> None:
    """v2 forks v1 and keeps most of its courses -- that must stay legal.

    This is the other half of the previous test. A constraint tightened to
    ``(course_id)`` alone would block every fork, which is the normal way a
    path is edited.
    """
    stage_in_v2 = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_path_stages (id, version_id, position, title) "
                "VALUES (:id, :version, 1, 'S')"
            ),
            {"id": stage_in_v2, "version": graph.version_two_id},
        )

    await _add_item(
        engine,
        version_id=graph.version_id,
        course_id=graph.course_a,
        stage_id=graph.stage_id,
        position=1,
    )
    await _add_item(
        engine,
        version_id=graph.version_two_id,
        course_id=graph.course_a,
        stage_id=stage_in_v2,
        position=1,
    )

    async with engine.begin() as conn:
        count = (
            await conn.execute(
                text(
                    "SELECT COUNT(*) FROM career_course_items "
                    "WHERE course_id = :course AND version_id = ANY(:versions)"
                ),
                {
                    "course": graph.course_a,
                    "versions": [graph.version_id, graph.version_two_id],
                },
            )
        ).scalar_one()
    assert count == 2


async def test_two_items_cannot_share_a_position_in_one_stage(
    engine: AsyncEngine, graph: Graph
) -> None:
    """Positions order the stage; a tie makes the route non-deterministic."""
    await _add_item(
        engine,
        version_id=graph.version_id,
        course_id=graph.course_a,
        stage_id=graph.stage_id,
        position=1,
    )
    await _expect_violation(
        engine,
        "career_course_items_stage_position_key",
        "INSERT INTO career_course_items (version_id, course_id, stage_id, position) "
        "VALUES (:version, :course, :stage, 1)",
        {
            "version": graph.version_id,
            "course": graph.course_b,
            "stage": graph.stage_id,
        },
    )


@pytest.mark.parametrize("position", [0, -1])
async def test_item_positions_are_positive(
    engine: AsyncEngine, graph: Graph, position: int
) -> None:
    await _expect_violation(
        engine,
        "career_course_items_position_check",
        "INSERT INTO career_course_items (version_id, course_id, stage_id, position) "
        "VALUES (:version, :course, :stage, :position)",
        {
            "version": graph.version_id,
            "course": graph.course_a,
            "stage": graph.stage_id,
            "position": position,
        },
    )


async def test_satisfied_by_accepts_only_completion(engine: AsyncEngine, graph: Graph) -> None:
    """``'pass'`` was removed in 0073; the evaluator is completion-only.

    A row carrying the retired value would be silently unsatisfiable -- the
    evaluator only ever looks for completion, so the course could never be
    ticked off and the stage would never unlock.
    """
    await _expect_violation(
        engine,
        "career_course_items_satisfied_by_check",
        "INSERT INTO career_course_items "
        "(version_id, course_id, stage_id, position, satisfied_by) "
        "VALUES (:version, :course, :stage, 1, 'pass')",
        {
            "version": graph.version_id,
            "course": graph.course_a,
            "stage": graph.stage_id,
        },
    )


# ---------------------------------------------------------------------------
# student_stage_progress -- the append-only latch
# ---------------------------------------------------------------------------


async def test_a_stage_latches_once_per_enrollment(engine: AsyncEngine, graph: Graph) -> None:
    """Re-running the evaluator must not stack duplicate latch rows.

    Stage completion is derived and recomputed often; without the unique key
    every recomputation would append another row and any COUNT over the table
    would drift upward on its own.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO student_stage_progress (id, enrollment_id, stage_id) "
                "VALUES (gen_random_uuid(), :enrollment, :stage)"
            ),
            {"enrollment": graph.enrollment_id, "stage": graph.stage_id},
        )

    await _expect_violation(
        engine,
        "student_stage_progress_enrollment_id_stage_id_key",
        "INSERT INTO student_stage_progress (id, enrollment_id, stage_id) "
        "VALUES (gen_random_uuid(), :enrollment, :stage)",
        {"enrollment": graph.enrollment_id, "stage": graph.stage_id},
    )


def test_the_latch_table_has_no_soft_delete_column() -> None:
    """The module's own claim, asserted rather than left as a comment.

    "Adding SoftDeleteMixin would let a soft-deleted latch row un-complete a
    student -- the exact failure this table exists to prevent." A future
    refactor that sweeps SoftDeleteMixin onto every model would reintroduce
    precisely that, and nothing else in the suite would notice.
    """
    columns = {c.key for c in inspect(StudentStageProgress).columns}
    assert "deleted_at" not in columns
    assert "deleted_by" not in columns
    # It is also not merely un-mapped: the append-only precedent is that these
    # rows carry a creation time and nothing that implies mutation.
    assert "updated_at" not in columns


def test_the_versioned_route_tables_do_carry_soft_delete() -> None:
    """The contrast that makes the previous test meaningful.

    Versions and stages are authored content and are soft-deleted; the latch
    is a fact about a student and is not. If both ended up the same, one of
    the two designs has been lost.
    """
    for model in (CareerPathVersion, CareerPathStage):
        columns = {c.key for c in inspect(model).columns}
        assert "deleted_at" in columns, model.__name__


def test_course_items_carry_no_soft_delete_either() -> None:
    """Documented in the model: the table is on the 0002 skip-list.

    The unique index on ``(stage_id, position)`` is NOT partial for this
    reason. If a ``deleted_at`` column reappeared, that index would start
    counting removed items and block their positions from being reused.
    """
    assert "deleted_at" not in {c.key for c in inspect(CareerPathCourse).columns}


# ---------------------------------------------------------------------------
# career_readiness_snapshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [-1, 101])
async def test_readiness_score_is_a_percentage(
    engine: AsyncEngine, graph: Graph, score: int
) -> None:
    """It is aggregated into pathway-level reporting, so an out-of-range value
    would skew every average silently rather than failing loudly.

    Just outside the bounds on each side, deliberately. A wildly wrong value
    like 1000 does not reach the CHECK at all -- the column is NUMERIC(5,2)
    and overflows first -- so it would exercise the column width rather than
    the constraint this test is named for.
    """
    await _expect_violation(
        engine,
        "career_readiness_snapshots_readiness_score_check",
        "INSERT INTO career_readiness_snapshots "
        "(id, student_id, career_path_id, version_id, readiness_score) "
        "VALUES (gen_random_uuid(), :student, :path, :version, :score)",
        {
            "student": graph.student_id,
            "path": graph.path_id,
            "version": graph.version_id,
            "score": score,
        },
    )


@pytest.mark.parametrize("score", [0, 50, 100])
async def test_the_readiness_bounds_are_inclusive(
    engine: AsyncEngine, graph: Graph, score: int
) -> None:
    """0 and 100 are both real, reachable answers, not off-by-one rejections."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_readiness_snapshots "
                "(id, student_id, career_path_id, version_id, readiness_score) "
                "VALUES (gen_random_uuid(), :student, :path, :version, :score)"
            ),
            {
                "student": graph.student_id,
                "path": graph.path_id,
                "version": graph.version_id,
                "score": score,
            },
        )


def test_the_snapshot_pins_a_version() -> None:
    """D3(a): an enrollment stays on the version it started on.

    A snapshot without the pin could not be compared over time -- the route it
    scored against would move underneath the series.
    """
    assert "version_id" in {c.key for c in inspect(CareerReadinessSnapshot).columns}
