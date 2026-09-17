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
from abridgeai.features.learning_programs.api import public as programs_api
from abridgeai.features.learning_programs.schemas import ProgramCreate, ProgramUpdate
from tests.support.db_graph import hard_delete_graph


def _async_url(url: str) -> str:
    return url.replace("+psycopg://", "+psycopg_async://")


@pytest.mark.asyncio
async def test_multiple_paths_complete_the_program_only_when_all_are_complete(
    engine: AsyncEngine,
    seeded_users: SeededUsers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())
    student = seeded_users.student_id

    async def two_path_limit(*args: object, **kwargs: object) -> int:
        return 2

    async def path_is_complete(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(services, "resolve_setting", two_path_limit)
    monkeypatch.setattr(
        programs_api.career_paths_api,
        "is_version_complete_for_user",
        path_is_complete,
    )

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"multi-path-{uuid.uuid4().hex[:8]}",
                name="Multi-path Program",
                max_career_paths_per_enrollment=2,
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        enrollment = (
            await services.enroll_students(
                db, program_id=program.id, student_ids=[student], actor=manager
            )
        )[0]

        assert program.current_version.max_career_paths_per_enrollment == 2
        assert enrollment.max_career_paths == 2

        second = await services.select_path(
            db, enrollment_id=enrollment.id, career_path_id=path_b, student_id=student
        )

        assert enrollment.selected_path_count == 1
        assert enrollment.attempts[0].selection_source == "program_default"
        assert second.selected_path_count == 2
        assert len([attempt for attempt in second.attempts if attempt.status == "active"]) == 2

        assert (
            await programs_api.complete_program_attempts(
                db, student_id=student, career_path_id=path_a
            )
            == 0
        )
        after_first = (await services.list_my_enrollments(db, student))[0]
        assert after_first.status == "active"
        assert sorted(attempt.status for attempt in after_first.attempts) == [
            "active",
            "completed",
        ]

        assert (
            await programs_api.complete_program_attempts(
                db, student_id=student, career_path_id=path_b
            )
            == 1
        )
        finished = (await services.list_my_enrollments(db, student))[0]
        assert finished.status == "completed"
        assert all(attempt.status == "completed" for attempt in finished.attempts)
        await db.rollback()


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


#: `(faculty_id, path_ids)` created by `_seed_program_context`, drained after
#: each test by the autouse cleanup below. The seeder is a plain helper called
#: mid-test rather than a fixture, so it has no teardown of its own.
_CREATED_CONTEXTS: list[tuple[uuid.UUID, list[uuid.UUID]]] = []


async def _seed_program_context(
    engine: AsyncEngine, seeded: SeededUsers
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    faculty_id, path_a, path_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    _CREATED_CONTEXTS.append((faculty_id, [path_a, path_b]))
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
        await conn.execute(
            text(
                "INSERT INTO user_faculty_assignments "
                "(id, user_id, organization_id, faculty_id, status) "
                "VALUES (gen_random_uuid(), :dean, :org, :faculty, 'active')"
            ),
            {
                "dean": seeded.hod_id,
                "org": seeded.organization_id,
                "faculty": faculty_id,
            },
        )
        # The manager operates on programs in this faculty too (create /
        # enrol / withdraw). `actor_has_program_role` accepts an org-scoped
        # role OR an org_unit-scoped one backed by a matching active faculty
        # assignment — conftest grants the manager the latter for ITS faculty,
        # not this one, so both rows are needed here as well.
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(id, user_id, role_id, scope_kind, organization_id, org_unit_id, granted_by) "
                "SELECT gen_random_uuid(), :mgr, id, 'org_unit', :org, :faculty, :mgr "
                "FROM roles WHERE code = 'manager' AND deleted_at IS NULL"
            ),
            {
                "mgr": seeded.manager_id,
                "org": seeded.organization_id,
                "faculty": faculty_id,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO user_faculty_assignments "
                "(id, user_id, organization_id, faculty_id, status) "
                "VALUES (gen_random_uuid(), :mgr, :org, :faculty, 'active')"
            ),
            {
                "mgr": seeded.manager_id,
                "org": seeded.organization_id,
                "faculty": faculty_id,
            },
        )
        for index, path_id in enumerate((path_a, path_b), start=1):
            await conn.execute(
                text(
                    "INSERT INTO career_paths "
                    "(id, organization_id, slug, name, status) "
                    "VALUES (:id, :org, :slug, :name, 'published')"
                ),
                {
                    "id": path_id,
                    "org": seeded.organization_id,
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


@pytest_asyncio.fixture(autouse=True)
async def _drain_program_contexts(engine: AsyncEngine) -> AsyncIterator[None]:
    """Remove everything `_seed_program_context` created, after every test."""
    yield
    while _CREATED_CONTEXTS:
        faculty_id, path_ids = _CREATED_CONTEXTS.pop()
        await _teardown_program_context(engine, faculty_id, path_ids)


async def _teardown_program_context(
    engine: AsyncEngine, faculty_id: uuid.UUID, path_ids: list[uuid.UUID]
) -> None:
    """Undo `_seed_program_context`.

    The grants it makes hang off SEEDED users (dean, manager), so leaving
    them behind inflates their assignment count for the rest of the session —
    tests/unit/test_fixtures.py::test_seed_users counts exactly that and saw
    17 where 5 were expected (5 + 6 tests x 2 grants).

    The org_unit and the two career paths hang off the SEEDED organization for
    the same reason, and used to survive: only the grants were removed. Their
    versions are graph-deleted rather than dropped by hand because
    `career_path_versions.career_path_id` is ON DELETE NO ACTION (migration
    0074) and the program tables reference the versions in turn.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM user_faculty_assignments WHERE faculty_id = :f"),
            {"f": faculty_id},
        )
        await conn.execute(
            text("DELETE FROM user_role_assignments WHERE org_unit_id = :f"),
            {"f": faculty_id},
        )
        await hard_delete_graph(conn, "career_paths", [str(p) for p in path_ids])
        await hard_delete_graph(conn, "org_units", [str(faculty_id)])


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
                slug=f"program-{uuid.uuid4().hex[:8]}",
                name="Versioned Program",
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        enrollments = await services.enroll_students(
            db, program_id=program.id, student_ids=[student], actor=manager
        )
        enrollment = enrollments[0]
        assert enrollment.status == "active"
        assert enrollment.attempts[-1].career_path_id == path_a
        assert enrollment.attempts[-1].selection_source == "program_default"

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
            decision_note="Keep focusing on the shared foundation courses.",
            actor=dean,
        )
        assert decided.status == "approved"
        assert decided.decision_note == "Keep focusing on the shared foundation courses."
        refreshed = (await services.list_my_enrollments(db, student))[0]
        assert refreshed.approved_switch_count == 1
        assert [attempt.status for attempt in refreshed.attempts] == ["switched_out", "active"]
        assert refreshed.attempts[0].exit_snapshot is not None
        assert refreshed.attempts[1].career_path_id == path_b
        await db.rollback()


@pytest.mark.asyncio
async def test_request_path_change_notifies_faculty_dean_with_deep_link(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Filing a path change request pushes an in-app notification to every
    owning Faculty Dean, deep-linking the targeted program's review tab."""
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())
    student = seeded_users.student_id

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                organization_id=seeded_users.organization_id,
                faculty_id=faculty_id,
                slug=f"program-{uuid.uuid4().hex[:8]}",
                name="Dean Notify Program",
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        enrollments = await services.enroll_students(
            db, program_id=program.id, student_ids=[student], actor=manager
        )
        enrollment = enrollments[0]
        request = await services.request_path_change(
            db,
            enrollment_id=enrollment.id,
            target_path_id=path_b,
            reason="Dean notify test",
            student_id=student,
        )
        await db.flush()

        rows = (
            (
                await db.execute(
                    text(
                        "SELECT category, entity_type, entity_id, action_url, title "
                        "FROM notifications WHERE user_id = :uid ORDER BY created_at DESC"
                    ),
                    {"uid": seeded_users.hod_id},
                )
            )
            .mappings()
            .all()
        )
        assert rows, "dean received no notification after a path change request"
        match = next(
            (r for r in rows if r["entity_id"] == request.id),
            None,
        )
        assert match is not None, f"no notification for request id {request.id}"
        assert match["category"] == "path_change_review"
        assert match["entity_type"] == "path_change_request"
        assert match["action_url"] == f"/management/learning-programs/{program.id}?tab=requests"
        assert "Path change request from" in match["title"]
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
                default_career_path_id=path_a,
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
                default_career_path_id=path_a,
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
async def test_new_program_version_requires_exactly_one_default_path(
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
                slug=f"default-required-{uuid.uuid4().hex[:8]}",
                name="Default path required",
                career_path_ids=[path_a, path_b],
            ),
            manager,
        )

        with pytest.raises(ConflictError, match="program_requires_exactly_one_default_path"):
            await services.publish_program(db, program_id=program.id, actor=manager)

        updated = await services.update_program(
            db,
            program_id=program.id,
            payload=ProgramUpdate(default_career_path_id=path_b),
            actor=manager,
        )
        assert [path.career_path_id for path in updated.paths if path.is_default] == [path_b]
        await services.publish_program(db, program_id=program.id, actor=manager)
        await db.rollback()


@pytest.mark.asyncio
async def test_default_path_must_be_replaced_before_removal(
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
                slug=f"replace-default-{uuid.uuid4().hex[:8]}",
                name="Replace default atomically",
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )

        with pytest.raises(ConflictError, match="default_path_must_be_replaced_before_removal"):
            await services.update_program(
                db,
                program_id=program.id,
                payload=ProgramUpdate(career_path_ids=[path_b]),
                actor=manager,
            )

        updated = await services.update_program(
            db,
            program_id=program.id,
            payload=ProgramUpdate(
                career_path_ids=[path_b],
                default_career_path_id=path_b,
            ),
            actor=manager,
        )
        assert len(updated.paths) == 1
        assert updated.paths[0].is_default is True
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
                default_career_path_id=path_a,
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
                default_career_path_id=path_b,
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
        with pytest.raises(ForbiddenError, match="primary_organization_required"):
            await services.create_program(
                db,
                ProgramCreate(
                    organization_id=seeded_users.organization_id,
                    faculty_id=faculty_id,
                    slug=f"admin-blocked-{uuid.uuid4().hex[:8]}",
                    name="Admin Must Not Create This",
                    career_path_ids=[path_a],
                    default_career_path_id=path_a,
                ),
                admin,
            )
        await db.rollback()


@pytest.mark.asyncio
async def test_program_list_cards_carry_dean_and_draft_stats(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    faculty_id, path_a, path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())
    student = seeded_users.student_id

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"list-cards-{uuid.uuid4().hex[:8]}",
                name="List Cards Program",
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        selected = (
            await services.enroll_students(
                db, program_id=program.id, student_ids=[student], actor=manager
            )
        )[0]
        await services.request_path_change(
            db,
            enrollment_id=selected.id,
            target_path_id=path_b,
            reason="Prefer the other path",
            student_id=student,
        )

        # Editing a published program opens a draft v2 (update_program creates
        # one), which is what has_draft_version surfaces.
        await services.update_program(
            db,
            program_id=program.id,
            payload=ProgramUpdate(description="revising"),
            actor=manager,
        )

        cards = await services.list_programs(
            db, organization_id=seeded_users.organization_id, actor=manager
        )
        card = next(c for c in cards if c.id == program.id)
        assert card.path_change_request_count == 1
        assert card.has_draft_version is True
        assert card.student_count == 1

        archived = await services.archive_program(db, program_id=program.id, actor=manager)
        assert archived.status == "archived"
        await db.rollback()


@pytest.mark.asyncio
async def test_approved_path_drop_ends_one_path_and_consumes_a_switch(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """A drop is a reviewed decision, not a self-service undo.

    It rides the same queue as a change and spends the same budget, so a
    student cannot drop-then-add their way around the dean.
    """
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
                slug=f"drop-{uuid.uuid4().hex[:8]}",
                name="Droppable Program",
                max_career_paths_per_enrollment=2,
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        enrollment = (
            await services.enroll_students(
                db, program_id=program.id, student_ids=[student], actor=manager
            )
        )[0]
        # Auto-enrolled onto the default, then the student adds the second.
        added = await services.select_path(
            db, enrollment_id=enrollment.id, career_path_id=path_b, student_id=student
        )
        assert added.selected_path_count == 2
        drop_target = next(
            attempt
            for attempt in added.attempts
            if attempt.career_path_id == path_b and attempt.status == "active"
        )

        request = await services.request_path_drop(
            db,
            enrollment_id=enrollment.id,
            from_attempt_id=drop_target.id,
            reason="I took on more than I can carry this semester",
            student_id=student,
        )
        assert request.kind == "drop"
        # A drop has no destination; the CHECK in migration 0122 is what keeps
        # that from reading as a switch with a missing target.
        assert request.target_career_path_id is None

        decided = await services.decide_change_request(
            db,
            request_id=request.id,
            approve=True,
            decision_reason=None,
            decision_note="Focus on the remaining path.",
            actor=dean,
        )
        assert decided.status == "approved"
        # No replacement attempt: this is what separates an approved drop from
        # an approved change in the student's own history.
        assert decided.new_attempt_id is None

        refreshed = (await services.list_my_enrollments(db, student))[0]
        assert refreshed.status == "active"
        dropped = next(row for row in refreshed.attempts if row.id == drop_target.id)
        assert dropped.status == "cancelled"
        assert dropped.ended_at is not None
        # Progress is kept, not erased — the same treatment a switch gets.
        assert dropped.exit_snapshot is not None
        remaining = [row for row in refreshed.attempts if row.status == "active"]
        assert [row.career_path_id for row in remaining] == [path_a]
        # Consumed, never refunded.
        assert refreshed.approved_switch_count == 1
        await db.rollback()


@pytest.mark.asyncio
async def test_single_path_program_selects_its_only_default_automatically(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    faculty_id, path_a, _path_b = await _seed_program_context(engine, seeded_users)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    manager = CurrentUser(seeded_users.manager_id, uuid.uuid4())

    async with factory() as db:
        program = await services.create_program(
            db,
            ProgramCreate(
                faculty_id=faculty_id,
                slug=f"single-default-{uuid.uuid4().hex[:8]}",
                name="Single path default",
                career_path_ids=[path_a],
            ),
            manager,
        )

        assert len(program.paths) == 1
        assert program.paths[0].career_path_id == path_a
        assert program.paths[0].is_default is True
        published = await services.publish_program(db, program_id=program.id, actor=manager)
        assert published.status == "published"
        await db.rollback()


@pytest.mark.asyncio
async def test_student_must_keep_at_least_one_active_path(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """Dropping the only path is leaving the program, which is a withdrawal.

    Checked when the request is filed AND again at approval, because the
    second path can end in between — an approved drop, a switch, or a
    completion — and an active enrolment with no active path is a state
    nothing else in the feature can produce or recover from.
    """
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
                slug=f"lastpath-{uuid.uuid4().hex[:8]}",
                name="Single Path Program",
                max_career_paths_per_enrollment=2,
                career_path_ids=[path_a, path_b],
                default_career_path_id=path_a,
            ),
            manager,
        )
        await services.publish_program(db, program_id=program.id, actor=manager)
        enrollment = (
            await services.enroll_students(
                db, program_id=program.id, student_ids=[student], actor=manager
            )
        )[0]
        only_attempt = enrollment.attempts[-1]

        # Filing is refused while it is the student's only active path.
        with pytest.raises(ConflictError, match="at_least_one_path_must_remain"):
            await services.request_path_drop(
                db,
                enrollment_id=enrollment.id,
                from_attempt_id=only_attempt.id,
                reason="I want out of this one",
                student_id=student,
            )

        # With a second path added the request is legal...
        added = await services.select_path(
            db, enrollment_id=enrollment.id, career_path_id=path_b, student_id=student
        )
        second = next(
            row
            for row in added.attempts
            if row.career_path_id == path_b and row.status == "active"
        )
        request = await services.request_path_drop(
            db,
            enrollment_id=enrollment.id,
            from_attempt_id=second.id,
            reason="Dropping the second path",
            student_id=student,
        )

        # ...until the OTHER path goes away before the dean decides. The
        # approval must re-check rather than trust the filing-time answer.
        await db.execute(
            text(
                "UPDATE program_path_attempts SET status = 'cancelled', ended_at = NOW() "
                "WHERE id = :attempt_id"
            ),
            {"attempt_id": only_attempt.id},
        )
        with pytest.raises(ConflictError, match="at_least_one_path_must_remain"):
            await services.decide_change_request(
                db,
                request_id=request.id,
                approve=True,
                decision_reason=None,
                decision_note=None,
                actor=dean,
            )
        await db.rollback()
