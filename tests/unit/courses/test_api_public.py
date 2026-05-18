"""Unit tests for ``features.courses.api.public`` (T20).

The unit suite is DB-free: it asserts the public surface is shaped
correctly (importable, signatures, DTO contract, re-exports). Behavior
that requires real rows (soft-delete filtering, trigger stamping) is
covered by integration tests in Wave 5 when consumers migrate onto
this module.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints
from uuid import UUID, uuid4

import pytest

from abridgeai.features.courses.api import public
from abridgeai.features.courses.api._dto import (
    ContentTreeDTO,
    ContentTreeItemDTO,
    CourseDTO,
    LessonDTO,
    ModuleDTO,
    ModuleItemDTO,
    OrgDTO,
)

_EXPECTED_FUNCTIONS = (
    "get_course_by_id",
    "get_lesson_by_id",
    "get_module_by_id",
    "walk_resource_to_course",
    "get_published_content_tree",
    "get_published_lessons_for_course",
    "find_module_items",
    "next_module_item_position",
    "insert_module_item",
    "get_lesson_title",
    "get_course_slug",
    "get_user_primary_org",
)

_EXPECTED_DTO_NAMES = (
    "CourseDTO",
    "LessonDTO",
    "ModuleDTO",
    "ModuleItemDTO",
    "ContentTreeDTO",
    "ContentTreeItemDTO",
    "OrgDTO",
)


def test_all_expected_symbols_in_dunder_all() -> None:
    exported = set(public.__all__)
    for name in _EXPECTED_FUNCTIONS:
        assert name in exported, f"{name} missing from __all__"
    for name in _EXPECTED_DTO_NAMES:
        assert name in exported, f"{name} missing from __all__"
    assert "require_lesson_authoring_access" in exported


def test_require_lesson_authoring_access_is_re_exported() -> None:
    from abridgeai.features.courses.routers._deps import (
        require_lesson_authoring_access as canonical,
    )

    assert public.require_lesson_authoring_access is canonical


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_function_is_async_coroutine(name: str) -> None:
    fn = getattr(public, name)
    assert inspect.iscoroutinefunction(fn), f"{name} must be `async def`"


@pytest.mark.parametrize("name", _EXPECTED_FUNCTIONS)
def test_first_positional_param_is_db(name: str) -> None:
    fn = getattr(public, name)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    assert params, f"{name} has no parameters"
    assert params[0].name == "db", (
        f"{name} first positional must be `db: AsyncSession`, got {params[0].name!r}"
    )


def test_get_course_by_id_signature() -> None:
    sig = inspect.signature(public.get_course_by_id)
    assert list(sig.parameters) == ["db", "course_id"]
    hints = get_type_hints(public.get_course_by_id)
    assert hints["course_id"] is UUID
    assert hints["return"] == (CourseDTO | None)


def test_walk_resource_to_course_signature() -> None:
    sig = inspect.signature(public.walk_resource_to_course)
    assert list(sig.parameters) == ["db", "resource_id"]
    hints = get_type_hints(public.walk_resource_to_course)
    assert hints["resource_id"] is UUID
    assert hints["return"] == (CourseDTO | None)


def test_find_module_items_is_keyword_only_lesson_id() -> None:
    sig = inspect.signature(public.find_module_items)
    params = sig.parameters
    assert params["lesson_id"].kind is inspect.Parameter.KEYWORD_ONLY


def test_insert_module_item_keyword_only_args() -> None:
    sig = inspect.signature(public.insert_module_item)
    kw_only = {
        name for name, p in sig.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    for required in ("module_id", "item_type", "position"):
        assert required in kw_only, f"{required} must be keyword-only"
    for optional in ("lesson_id", "quiz_id", "interview_config_id"):
        assert sig.parameters[optional].default is None


def test_next_module_item_position_returns_int() -> None:
    hints = get_type_hints(public.next_module_item_position)
    assert hints["return"] is int


def test_dtos_are_frozen_and_from_attributes() -> None:
    for dto in (
        CourseDTO,
        LessonDTO,
        ModuleDTO,
        ModuleItemDTO,
        ContentTreeDTO,
        ContentTreeItemDTO,
        OrgDTO,
    ):
        cfg = dto.model_config
        assert cfg.get("frozen") is True, f"{dto.__name__} must be frozen"
        assert cfg.get("from_attributes") is True, (
            f"{dto.__name__} must allow ORM attribute hydration"
        )


def test_course_dto_has_only_public_fields() -> None:
    fields = set(CourseDTO.model_fields)
    assert fields == {
        "id",
        "organization_id",
        "org_unit_id",
        "owner_user_id",
        "slug",
        "title",
        "status",
    }
    for forbidden in ("created_by", "updated_by", "deleted_at", "deleted_by"):
        assert forbidden not in fields, f"{forbidden} must not leak into the cross-feature contract"


def test_lesson_dto_has_only_public_fields() -> None:
    fields = set(LessonDTO.model_fields)
    assert fields == {"id", "module_id", "slug", "title", "status"}


def test_module_item_dto_polymorphism_fields() -> None:
    fields = set(ModuleItemDTO.model_fields)
    assert {"lesson_id", "quiz_id", "interview_config_id", "item_type"} <= fields


def test_org_dto_minimal_shape() -> None:
    assert set(OrgDTO.model_fields) == {"id"}


def test_content_tree_dto_round_trip() -> None:
    course_id = uuid4()
    module_id = uuid4()
    lesson_id = uuid4()
    org_id = uuid4()
    owner_id = uuid4()
    payload = {
        "course": {
            "id": course_id,
            "organization_id": org_id,
            "org_unit_id": None,
            "owner_user_id": owner_id,
            "slug": "intro",
            "title": "Intro",
            "status": "published",
        },
        "modules": [
            {
                "id": module_id,
                "course_id": course_id,
                "title": "Mod",
                "position": 1,
                "status": "published",
            }
        ],
        "items": [
            {
                "id": uuid4(),
                "module_id": module_id,
                "item_type": "lesson",
                "lesson_id": lesson_id,
                "quiz_id": None,
                "interview_config_id": None,
                "position": 1,
                "lesson": {
                    "id": lesson_id,
                    "module_id": module_id,
                    "slug": "l1",
                    "title": "Lesson 1",
                    "status": "published",
                },
            }
        ],
    }
    tree = ContentTreeDTO.model_validate(payload)
    assert tree.course.id == course_id
    assert tree.modules[0].id == module_id
    assert tree.items[0].lesson is not None
    assert tree.items[0].lesson.id == lesson_id


def test_content_tree_dto_drops_extra_fields() -> None:
    course_id = uuid4()
    org_id = uuid4()
    owner_id = uuid4()
    payload = {
        "course": {
            "id": course_id,
            "organization_id": org_id,
            "org_unit_id": None,
            "owner_user_id": owner_id,
            "slug": "intro",
            "title": "Intro",
            "status": "published",
            "description": "ignored on purpose",
            "created_by": uuid4(),
        },
        "modules": [],
        "items": [],
    }
    tree = ContentTreeDTO.model_validate(payload)
    assert not hasattr(tree.course, "description")
    assert not hasattr(tree.course, "created_by")


def test_dtos_are_immutable() -> None:
    org = OrgDTO(id=uuid4())
    with pytest.raises((TypeError, ValueError)):
        org.id = uuid4()  # type: ignore[misc]


def test_module_docstring_documents_public_contract() -> None:
    doc = (public.__doc__ or "").lower()
    assert "cross-feature" in doc
    assert "soft-delete" in doc


from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession  # noqa: E402

from abridgeai.features.courses.api.public import (  # noqa: E402
    get_published_lessons_for_course,
)


async def _seed_course_tree(
    session: AsyncSession,
    *,
    org_id: UUID,
    owner_id: UUID,
    course_id: UUID,
    course_status: str = "draft",
    course_deleted: bool = False,
    module_status: str = "published",
    lesson_specs: tuple[tuple[UUID, str], ...] = (),
) -> None:
    """``lesson_specs`` is a tuple of ``(lesson_id, lesson_status)`` pairs;
    each is also linked via a ``module_items`` row of ``item_type='lesson'``.
    """
    suffix = course_id.hex[:8]
    await session.execute(
        text(
            "INSERT INTO organizations (id, slug, name, status) "
            "VALUES (:id, :slug, :name, 'active') ON CONFLICT DO NOTHING"
        ),
        {"id": str(org_id), "slug": f"o-{suffix}", "name": "O"},
    )
    await session.execute(
        text("INSERT INTO users (id, primary_email) VALUES (:id, :email) ON CONFLICT DO NOTHING"),
        {"id": str(owner_id), "email": f"u-{suffix}@e.com"},
    )
    deleted_clause = "NOW()" if course_deleted else "NULL"
    await session.execute(
        text(
            "INSERT INTO courses "  # noqa: S608  # test fixture: deleted_clause is a code-controlled literal
            "(id, organization_id, owner_user_id, slug, title, status, deleted_at) "
            f"VALUES (:id, :org, :owner, :slug, :title, :status, {deleted_clause})"
        ),
        {
            "id": str(course_id),
            "org": str(org_id),
            "owner": str(owner_id),
            "slug": f"c-{suffix}",
            "title": "C",
            "status": course_status,
        },
    )
    module_id = uuid4()
    await session.execute(
        text(
            "INSERT INTO modules (id, course_id, title, position, status) "
            "VALUES (:id, :c, :title, 1, :status)"
        ),
        {
            "id": str(module_id),
            "c": str(course_id),
            "title": "M",
            "status": module_status,
        },
    )
    for idx, (lesson_id, lesson_status) in enumerate(lesson_specs, start=1):
        await session.execute(
            text(
                "INSERT INTO lessons "
                "(id, module_id, slug, title, status) "
                "VALUES (:id, :m, :slug, :title, :status)"
            ),
            {
                "id": str(lesson_id),
                "m": str(module_id),
                "slug": f"l-{lesson_id.hex[:8]}",
                "title": f"L{idx}",
                "status": lesson_status,
            },
        )
        await session.execute(
            text(
                "INSERT INTO module_items "
                "(id, module_id, item_type, lesson_id, position) "
                "VALUES (:id, :m, 'lesson', :l, :pos)"
            ),
            {
                "id": str(uuid4()),
                "m": str(module_id),
                "l": str(lesson_id),
                "pos": idx,
            },
        )


@pytest.mark.asyncio
async def test_get_published_lessons_for_course_returns_lessons_in_draft_course(
    test_engine: AsyncEngine,
) -> None:
    org_id = uuid4()
    owner_id = uuid4()
    course_id = uuid4()
    lesson_a = uuid4()
    lesson_b = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_course_tree(
                session,
                org_id=org_id,
                owner_id=owner_id,
                course_id=course_id,
                course_status="draft",
                lesson_specs=((lesson_a, "published"), (lesson_b, "published")),
            )
            await session.flush()

            lessons = await get_published_lessons_for_course(session, course_id)
        finally:
            await trans.rollback()

    returned_ids = {lesson.id for lesson in lessons}
    assert returned_ids == {lesson_a, lesson_b}
    for lesson in lessons:
        assert isinstance(lesson, LessonDTO)
        assert lesson.status == "published"


@pytest.mark.asyncio
async def test_get_published_lessons_for_course_excludes_non_published(
    test_engine: AsyncEngine,
) -> None:
    org_id = uuid4()
    owner_id = uuid4()
    course_id = uuid4()
    pub_lesson = uuid4()
    draft_lesson = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_course_tree(
                session,
                org_id=org_id,
                owner_id=owner_id,
                course_id=course_id,
                course_status="published",
                lesson_specs=(
                    (pub_lesson, "published"),
                    (draft_lesson, "draft"),
                ),
            )
            await session.flush()

            lessons = await get_published_lessons_for_course(session, course_id)
        finally:
            await trans.rollback()

    returned_ids = {lesson.id for lesson in lessons}
    assert returned_ids == {pub_lesson}


@pytest.mark.asyncio
async def test_get_published_lessons_for_course_soft_deleted_course_returns_empty(
    test_engine: AsyncEngine,
) -> None:
    org_id = uuid4()
    owner_id = uuid4()
    course_id = uuid4()
    lesson_id = uuid4()

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await _seed_course_tree(
                session,
                org_id=org_id,
                owner_id=owner_id,
                course_id=course_id,
                course_status="published",
                course_deleted=True,
                lesson_specs=((lesson_id, "published"),),
            )
            await session.flush()

            lessons = await get_published_lessons_for_course(session, course_id)
        finally:
            await trans.rollback()

    assert lessons == []
