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


def test_course_faculty_is_chosen_at_creation_and_reassignable() -> None:
    """Faculty is picked at creation AND changeable afterwards.

    This test previously asserted ``"faculty_id" not in CourseUpdate.model_fields``
    — faculty was write-once. That was reversed deliberately (user request
    2026-09-05): every course predating the faculty feature sat at NULL with no
    route to fix it, so a "filter courses by faculty" view could never match
    them. Reassignment is manager-only and tenancy-checked in
    ``courses.services.authoring._validate_faculty_reassignment``; see
    ``tests/integration/test_course_faculty_reassignment.py``.

    ``org_unit_id`` stays absent from both: migration 0094 renamed the column to
    ``faculty_id`` and flattened the tree, so accepting the old name would let a
    caller address a concept the schema no longer has.
    """
    faculty_id = uuid4()
    create = CourseCreate(title="Algorithms", slug="algorithms", faculty_id=faculty_id)

    assert create.faculty_id == faculty_id
    assert "faculty_id" in CourseUpdate.model_fields
    assert "org_unit_id" not in CourseCreate.model_fields
    assert "org_unit_id" not in CourseUpdate.model_fields


def test_career_path_is_organization_wide() -> None:
    fields = CareerPathCreate.model_fields

    assert "org_unit_id" not in fields
    assert "faculty_id" not in fields
