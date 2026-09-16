from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.courses.api import public


class _DTO:
    @staticmethod
    def model_validate(value: object) -> tuple[str, object]:
        return ("dto", value)


@pytest.mark.asyncio
async def test_thumbnail_url_wrapper_uses_catalog_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from abridgeai.features.courses.services import catalog as catalog_service

    course_ids = [uuid4()]
    expected = {course_ids[0]: "https://storage/thumbnail"}
    lookup = AsyncMock(return_value=expected)
    monkeypatch.setattr(catalog_service, "get_course_thumbnail_urls", lookup)

    assert await public.get_course_thumbnail_urls(object(), course_ids) == expected
    lookup.assert_awaited_once()


@pytest.mark.asyncio
async def test_optional_single_resource_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    course, lesson, module = object(), object(), object()
    monkeypatch.setattr(public, "CourseDTO", _DTO)
    monkeypatch.setattr(public, "LessonDTO", _DTO)
    monkeypatch.setattr(public, "ModuleDTO", _DTO)
    monkeypatch.setattr(public.queries, "get_course", AsyncMock(side_effect=[course, None]))
    monkeypatch.setattr(public.queries, "get_lesson", AsyncMock(side_effect=[lesson, None]))
    monkeypatch.setattr(public.queries, "get_module", AsyncMock(side_effect=[module, None]))

    assert await public.get_course_by_id(object(), uuid4()) == ("dto", course)
    assert await public.get_course_by_id(object(), uuid4()) is None
    assert await public.get_lesson_by_id(object(), uuid4()) == ("dto", lesson)
    assert await public.get_lesson_by_id(object(), uuid4()) is None
    assert await public.get_module_by_id(object(), uuid4()) == ("dto", module)
    assert await public.get_module_by_id(object(), uuid4()) is None


@pytest.mark.asyncio
async def test_course_list_and_scalar_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public, "CourseDTO", _DTO)
    monkeypatch.setattr(public, "LessonDTO", _DTO)
    monkeypatch.setattr(public, "ModuleItemDTO", _DTO)
    rows = [SimpleNamespace(outcome_text="Outcome A"), SimpleNamespace(outcome_text="Outcome B")]
    monkeypatch.setattr(public.queries, "list_courses_by_org", AsyncMock(return_value=[1, 2]))
    monkeypatch.setattr(public.queries, "list_course_outcomes", AsyncMock(return_value=rows))
    monkeypatch.setattr(
        public.queries, "get_published_lessons_for_course", AsyncMock(return_value=[3])
    )
    monkeypatch.setattr(public.queries, "find_module_items_by_lesson", AsyncMock(return_value=[4]))
    monkeypatch.setattr(public.queries, "list_courses_for_teacher", AsyncMock(return_value=[5]))
    lesson_ids = [uuid4()]
    monkeypatch.setattr(
        public.queries, "list_lesson_ids_for_modules", AsyncMock(return_value=lesson_ids)
    )
    monkeypatch.setattr(public.queries, "next_module_item_position", AsyncMock(return_value=7))
    monkeypatch.setattr(public.queries, "get_lesson_title", AsyncMock(return_value="Lesson"))
    monkeypatch.setattr(public.queries, "get_course_slug", AsyncMock(return_value="course"))

    db = object()
    assert await public.list_courses_by_org(db, uuid4()) == [("dto", 1), ("dto", 2)]
    assert await public.list_course_outcome_texts(db, uuid4()) == ["Outcome A", "Outcome B"]
    assert await public.get_published_lessons_for_course(db, uuid4()) == [("dto", 3)]
    assert await public.find_module_items(db, lesson_id=uuid4()) == [("dto", 4)]
    assert await public.list_courses_for_teacher(db, uuid4()) == [("dto", 5)]
    assert await public.list_lesson_ids_for_modules(db, [uuid4()]) == lesson_ids
    assert await public.next_module_item_position(db, uuid4()) == 7
    assert await public.get_lesson_title(db, uuid4()) == "Lesson"
    assert await public.get_course_slug(db, uuid4()) == "course"


@pytest.mark.asyncio
async def test_course_manager_ids_handles_missing_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from abridgeai.features.courses.queries import assignment as assignment_queries

    owner, teacher = uuid4(), uuid4()
    monkeypatch.setattr(public.queries, "get_course", AsyncMock(side_effect=[None, SimpleNamespace(owner_user_id=owner)]))
    monkeypatch.setattr(
        assignment_queries,
        "list_teachers_for_course",
        AsyncMock(return_value=[{"user_id": owner}, {"user_id": teacher}]),
    )

    assert await public.list_course_manager_ids(object(), uuid4()) == []
    assert set(await public.list_course_manager_ids(object(), uuid4())) == {owner, teacher}


@pytest.mark.asyncio
async def test_walk_content_tree_and_insert_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public, "CourseDTO", _DTO)
    monkeypatch.setattr(public, "ContentTreeDTO", _DTO)
    monkeypatch.setattr(public, "ContentTreeItemDTO", _DTO)
    monkeypatch.setattr(public, "ModuleItemDTO", _DTO)
    monkeypatch.setattr(public.queries, "walk_resource_to_course", AsyncMock(side_effect=[None, "course-row"]))
    monkeypatch.setattr(public, "get_published_course_content", AsyncMock(side_effect=[None, {"course": "c", "modules": ["m"], "items": ["i"]}]))
    insert = AsyncMock(return_value="inserted")
    monkeypatch.setattr(public.queries, "insert_module_item", insert)

    assert await public.walk_resource_to_course(object(), uuid4()) is None
    assert await public.walk_resource_to_course(object(), uuid4()) == ("dto", "course-row")
    assert await public.get_published_content_tree(object(), uuid4()) is None
    tree = await public.get_published_content_tree(object(), uuid4())
    assert tree[0] == "dto"
    assert tree[1]["items"] == [("dto", "i")]

    item = await public.insert_module_item(
        object(), module_id=uuid4(), item_type="lesson", position=2, lesson_id=uuid4()
    )
    assert item == ("dto", "inserted")
    assert insert.await_args.kwargs["position"] == 2


@pytest.mark.asyncio
async def test_primary_org_and_content_access_decision_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from abridgeai.features.access_control import policies
    from abridgeai.features.access_control.api import public as access_api
    from abridgeai.features.enrollments.api import public as enrollments_api

    monkeypatch.setattr(public.queries, "get_user_primary_org_id", AsyncMock(side_effect=[None, uuid4()]))
    assert await public.get_user_primary_org(object(), uuid4()) is None
    assert (await public.get_user_primary_org(object(), uuid4())).id is not None

    manage = AsyncMock(side_effect=[True, False, False, False])
    monkeypatch.setattr(policies, "can_manage_course", manage)
    monkeypatch.setattr(public.queries, "get_course_org", AsyncMock(side_effect=[None, uuid4(), uuid4()]))
    monkeypatch.setattr(access_api, "is_user_member_of_org", AsyncMock(side_effect=[False, True]))
    monkeypatch.setattr(
        enrollments_api,
        "has_active_or_completed_enrollment",
        AsyncMock(return_value=True),
    )

    db, user_id, course_id = object(), uuid4(), uuid4()
    assert await public.can_view_course_content(db, user_id=user_id, course_id=course_id)
    assert not await public.can_view_course_content(db, user_id=user_id, course_id=course_id)
    assert not await public.can_view_course_content(db, user_id=user_id, course_id=course_id)
    assert await public.can_view_course_content(db, user_id=user_id, course_id=course_id)
