from __future__ import annotations

from uuid import uuid4

from abridgeai.features.access_control.schemas.admin import OrgUnitCreate, OrgUnitPatch
from abridgeai.features.career_paths.schemas.authoring import CareerPathCreate
from abridgeai.features.courses.schemas.request import CourseCreate, CourseUpdate


def test_faculty_is_the_only_creatable_organization_unit() -> None:
    payload = OrgUnitCreate(name="Engineering", code="ENG")

    assert payload.unit_type == "faculty"
    assert "parent_unit_id" not in OrgUnitCreate.model_fields
    assert "parent_unit_id" not in OrgUnitPatch.model_fields
    assert "unit_type" not in OrgUnitPatch.model_fields


def test_course_faculty_is_selected_only_at_creation() -> None:
    faculty_id = uuid4()
    create = CourseCreate(title="Algorithms", slug="algorithms", faculty_id=faculty_id)

    assert create.faculty_id == faculty_id
    assert "faculty_id" not in CourseUpdate.model_fields
    assert "org_unit_id" not in CourseCreate.model_fields
    assert "org_unit_id" not in CourseUpdate.model_fields


def test_career_path_is_organization_wide() -> None:
    fields = CareerPathCreate.model_fields

    assert "org_unit_id" not in fields
    assert "faculty_id" not in fields
