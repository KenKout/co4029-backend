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
from abridgeai.core.exceptions import ForbiddenError
from abridgeai.core.security import CurrentUser
from abridgeai.features.learning_programs import services
from abridgeai.features.learning_programs.schemas import ProgramCreate


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
