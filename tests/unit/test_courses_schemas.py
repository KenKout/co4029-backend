from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from abridgeai.features.courses.schemas import (
    CourseAuthoring,
    CourseContentPublic,
    CourseCreate,
    CoursePublic,
    InstructorAuthoring,
    InstructorRead,
    LessonAuthoring,
    LessonCreate,
    LessonPublic,
    LessonResourcePublic,
    ModuleAuthoring,
    ModuleCreate,
    ModulePublic,
    TagAuthoring,
    TagPublic,
)

_PUBLIC_FORBIDDEN = frozenset(
    {
        "internal_notes",
        "draft_notes",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
        "primary_email",
    }
)


def test_authoring_inherits_public() -> None:
    assert issubclass(CourseAuthoring, CoursePublic)
    assert issubclass(LessonAuthoring, LessonPublic)
    assert issubclass(ModuleAuthoring, ModulePublic)
    assert issubclass(TagAuthoring, TagPublic)
    assert issubclass(InstructorAuthoring, InstructorRead)


def test_public_excludes_internal_fields() -> None:
    public_schemas = (
        CoursePublic,
        ModulePublic,
        LessonPublic,
        LessonResourcePublic,
        InstructorRead,
        TagPublic,
    )
    for schema in public_schemas:
        leaked = _PUBLIC_FORBIDDEN & set(schema.model_fields.keys())
        assert not leaked, f"{schema.__name__} leaks internal fields: {leaked}"


def test_authoring_includes_authoring_fields() -> None:
    course_fields = set(CourseAuthoring.model_fields.keys())
    for required in ("created_by", "updated_by", "deleted_at", "owner_user_id"):
        assert required in course_fields, f"CourseAuthoring missing {required}"

    lesson_fields = set(LessonAuthoring.model_fields.keys())
    for required in (
        "ef_min_unlock",
        "tau_unlock",
        "requires_interview_pass",
        "unlock_rule_json",
        "prereq_lesson_ids",
        "deleted_at",
    ):
        assert required in lesson_fields, f"LessonAuthoring missing {required}"


def test_orm_compat_course_public() -> None:
    instructor = SimpleNamespace(
        user_id=uuid4(),
        display_name="Dr. Ada",
        avatar_url=None,
        headline="Algorithms",
    )
    course_row = SimpleNamespace(
        id=uuid4(),
        slug="intro-algos",
        title="Intro to Algorithms",
        description="CLRS warm-up",
        organization_id=uuid4(),
        instructor=instructor,
        status="published",
        tags=[],
        outcomes=[],
    )
    public = CoursePublic.model_validate(course_row)
    assert public.title == "Intro to Algorithms"
    assert public.instructor is not None
    assert public.instructor.display_name == "Dr. Ada"


def test_orm_compat_lesson_authoring() -> None:
    now = datetime.now(UTC)
    lesson_row = SimpleNamespace(
        id=uuid4(),
        title="Sorting in O(n log n)",
        module_id=uuid4(),
        slug="sorting",
        summary="Mergesort + quicksort",
        notes_markdown=None,
        primary_material_id=None,
        lesson_type="video",
        difficulty=None,
        estimated_minutes=30,
        status="draft",
        ef_min_unlock=2.0,
        tau_unlock=0.8,
        requires_interview_pass=False,
        unlock_rule_json={},
        prereq_lesson_ids=[],
        created_by=None,
        updated_by=None,
        created_at=now,
        updated_at=now,
        deleted_at=None,
        deleted_by=None,
    )
    authoring = LessonAuthoring.model_validate(lesson_row)
    assert authoring.ef_min_unlock == 2.0
    assert authoring.requires_interview_pass is False


def test_status_literal_narrowing() -> None:
    course_id = uuid4()
    org_id = uuid4()
    with pytest.raises(ValidationError):
        CoursePublic(
            id=course_id,
            slug="x",
            title="Y",
            organization_id=org_id,
            status="draft",  # type: ignore[arg-type]
        )

    now = datetime.now(UTC)
    authoring = CourseAuthoring(
        id=course_id,
        slug="x",
        title="Y",
        organization_id=org_id,
        owner_user_id=uuid4(),
        status="draft",
        created_at=now,
        updated_at=now,
    )
    assert authoring.status == "draft"


def test_lesson_create_validators() -> None:
    module_id = uuid4()

    with pytest.raises(ValidationError):
        LessonCreate(
            module_id=module_id,
            slug="too-easy",
            title="T",
            ef_min_unlock=1.0,
            tau_unlock=0.8,
        )

    ok = LessonCreate(
        module_id=module_id,
        slug="ok",
        title="T",
        ef_min_unlock=2.0,
        tau_unlock=0.8,
    )
    assert ok.ef_min_unlock == 2.0

    with pytest.raises(ValidationError):
        LessonCreate(
            module_id=module_id,
            slug="zero-tau",
            title="T",
            ef_min_unlock=2.0,
            tau_unlock=0.0,
        )


def test_course_content_public_no_leak() -> None:
    course = CoursePublic(
        id=uuid4(),
        slug="leak-check",
        title="Leak Check",
        organization_id=uuid4(),
        status="published",
    )
    content = CourseContentPublic(course=course, modules=[])
    dumped = content.model_dump_json()
    for forbidden in _PUBLIC_FORBIDDEN:
        assert forbidden not in dumped, f"{forbidden} leaked into JSON dump"


def test_request_strict_extras_rejected() -> None:
    with pytest.raises(ValidationError):
        CourseCreate(
            slug="s",
            title="t",
            sneaky_field="nope",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        ModuleCreate(
            course_id=uuid4(),
            title="t",
            position=0,
            unexpected="x",  # type: ignore[call-arg]
        )
