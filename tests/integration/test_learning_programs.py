from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from conftest import SeededUsers
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import ConflictError, ForbiddenError
from abridgeai.core.security import CurrentUser
from abridgeai.features.learning_programs import services
from abridgeai.features.learning_programs.schemas import ProgramCreate, ProgramUpdate


def _async_url(url: str) -> str:
    return url.replace("+psycopg://", "+psycopg_async://")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:  # noqa: ASYNC240
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]  # noqa: ASYNC240
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(config, "head")
    value = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield value
    await value.dispose()


async def _seed_program_context(
    engine: AsyncEngine, seeded: SeededUsers
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    faculty_id, path_a, path_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_units "
                "(id, organization_id, unit_type, name, code) "
                "VALUES (:id, :org, 'faculty', 'Program Faculty', :code)"
            ),
            {"id": faculty_id, "org": seeded.organization_id, "code": uuid.uuid4().hex[:8]},
        )
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, org_unit_id, granted_by) "
                "SELECT gen_random_uuid(), :dean, id, 'org_unit', :org, :faculty, :manager "
                "FROM roles WHERE code = 'hod' AND deleted_at IS NULL"
            ),
            {
                "dean": seeded.hod_id,
                "org": seeded.organization_id,
                "faculty": faculty_id,
                "manager": seeded.manager_id,
            },
        )
        for index, path_id in enumerate((path_a, path_b), start=1):
            await conn.execute(
                text(
                    "INSERT INTO career_paths "
                    "(id, organization_id, org_unit_id, slug, name, status) "
                    "VALUES (:id, :org, :faculty, :slug, :name, 'published')"
                ),
                {
                    "id": path_id,
                    "org": seeded.organization_id,
                    "faculty": faculty_id,
                    "slug": f"program-path-{uuid.uuid4().hex[:8]}",
                    "name": f"Program Path {index}",
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO career_path_versions "
                    "(id, career_path_id, version_no, status, published_at) "
                    "VALUES (gen_random_uuid(), :path, 1, 'published', NOW())"
                ),
                {"path": path_id},
            )
    return faculty_id, path_a, path_b


@pytest.mark.asyncio
async def test_program_selection_and_dean_approved_switch_are_historical(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())
    student = seeded_users.student_id
    dean = CurrentUser(seeded_users.hod_id, uuid.uuid4())

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                organization_id=seeded_users.organization_id,
                faculty_id=faculty_id,
                owner_faculty_dean_id=seeded_users.hod_id,
                slug=f"program-{uuid.uuid4().hex[:8]}",
                name="Versioned Program",
                career_path_ids=[path_a, path_b],
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        enrollments = await services.enroll_students(
            db, program_id=program.id, student_ids=[student], actor=manager
        )
        enrollment = enrollments[0]
        assert enrollment.status == "awaiting_path"

        selected = await services.select_path(
            db, enrollment_id=enrollment.id, career_path_id=path_a, student_id=student
        )
        assert selected.status == "active"
        assert selected.attempts[-1].career_path_id == path_a

        request = await services.request_path_change(
            db,
            enrollment_id=enrollment.id,
            target_path_id=path_b,
            reason="The target path better matches my plan",
            student_id=student,
        )
        decided = await services.decide_change_request(
            db,
            request_id=request.id,
            approve=True,
            decision_reason="Approved",
            actor=dean,
        )
        assert decided.status == "approved"
        refreshed = (await services.list_my_enrollments(db, student))[0]
        assert refreshed.approved_switch_count == 1
        assert [attempt.status for attempt in refreshed.attempts] == ["switched_out", "active"]
        assert refreshed.attempts[0].exit_snapshot is not None
        assert refreshed.attempts[1].career_path_id == path_b
        await db.rollback()


@pytest.mark.asyncio
async def test_removing_path_from_new_draft_preserves_pinned_versions_and_old_enrollment(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"remove-path-{uuid.uuid4().hex[:8]}",
                name="Path removal preserves history",
                career_path_ids=[path_a, path_b],
            ),
            manager,
        )
        published_v1 = await services.publish_program(db, program_id=program.id, actor=manager)
        enrollment = (
            await services.enroll_students(
                db,
                program_id=program.id,
                student_ids=[seeded_users.student_id],
                actor=manager,
            )
        )[0]

        # A newer Career Path version exists before the Program draft is edited.
        # Removing B must not silently upgrade retained path A from v1 to v2.
        await db.execute(
            text(
                "INSERT INTO career_path_versions "
                "(id, career_path_id, version_no, status, published_at) "
                "VALUES (gen_random_uuid(), :path, 2, 'published', NOW())"
            ),
            {"path": path_a},
        )

        draft_v2 = await services.update_program(
            db,
            program_id=program.id,
            payload=ProgramUpdate(career_path_ids=[path_a]),
            actor=manager,
        )

        assert draft_v2.current_version.version_no == 2
        assert [path.career_path_id for path in draft_v2.paths] == [path_a]
        assert draft_v2.paths[0].career_path_version_no == 1

        published_v2 = await services.publish_program(db, program_id=program.id, actor=manager)
        assert [path.career_path_id for path in published_v2.paths] == [path_a]

        old_enrollment = (await services.list_my_enrollments(db, seeded_users.student_id))[0]
        assert enrollment.program_version_id == published_v1.current_version.id
        assert old_enrollment.program_version_id == published_v1.current_version.id
        assert [path.career_path_id for path in old_enrollment.paths] == [path_a, path_b]
        await db.rollback()


@pytest.mark.asyncio
async def test_publish_rejects_path_archived_after_draft_was_created(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"archive-before-publish-{uuid.uuid4().hex[:8]}",
                name="Archived path publish guard",
                career_path_ids=[path_a, path_b],
            ),
            manager,
        )
        await db.execute(
            text("UPDATE career_paths SET status = 'archived' WHERE id = :path"),
            {"path": path_b},
        )

        with pytest.raises(ConflictError, match="program_contains_unavailable_paths"):
            await services.publish_program(db, program_id=program.id, actor=manager)
        await db.rollback()


@pytest.mark.asyncio
async def test_concurrency_cap_conflict_is_human_readable(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Hitting the concurrent-program cap must explain itself, not emit a code.

    The manager used to see the raw
    ``concurrent_program_limit_reached:<uuid>:1`` in a toast: no name, no
    number they could act on. The service now raises
    :class:`services.ProgramConflictError`, whose ``code`` stays stable for
    FE branching while ``message`` carries the sentence and ``fields`` the
    ids/limits.
    """
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())
    student = seeded_users.student_id

    async with factory() as db:
        # The org cap is 1 by default (settings_registry:
        # learning_program.max_concurrent_enrollments), so one live
        # enrollment is enough to make the second one collide.
        first = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"cap-first-{uuid.uuid4().hex[:8]}",
                name="Cap Program One",
                career_path_ids=[path_a],
            ),
            manager,
        )
        await services.publish_program(db, program_id=first.id, actor=manager)
        await services.enroll_students(
            db, program_id=first.id, student_ids=[student], actor=manager
        )

        second = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"cap-second-{uuid.uuid4().hex[:8]}",
                name="Cap Program Two",
                career_path_ids=[path_b],
            ),
            manager,
        )
        await services.publish_program(db, program_id=second.id, actor=manager)

        with pytest.raises(services.ProgramConflictError) as caught:
            await services.enroll_students(
                db, program_id=second.id, student_ids=[student], actor=manager
            )

        exc = caught.value
        assert exc.code == "concurrent_program_limit_reached"
        # No raw uuid, no colon-packed code — a sentence naming the student.
        assert str(student) not in exc.message
        assert "concurrent_program_limit_reached" not in exc.message
        assert "learning program" in exc.message
        assert exc.fields["student_id"] == str(student)
        assert exc.fields["limit"] == 1
        assert exc.fields["current"] == 1
        await db.rollback()


@pytest.mark.asyncio
async def test_it_admin_cannot_operate_academic_programs(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    faculty_id, path_a, _path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    admin = CurrentUser(seeded_users.admin_id, uuid.uuid4())
    async with factory() as db:
        with pytest.raises(ForbiddenError, match="manager_or_faculty_dean_scope_required"):
            await services.create_program(
                db,
                ProgramCreate(
                    organization_id=seeded_users.organization_id,
                    faculty_id=faculty_id,
                    owner_faculty_dean_id=seeded_users.hod_id,
                    slug=f"admin-blocked-{uuid.uuid4().hex[:8]}",
                    name="Admin Must Not Create This",
                    career_path_ids=[path_a],
                ),
                admin,
            )
        await db.rollback()
