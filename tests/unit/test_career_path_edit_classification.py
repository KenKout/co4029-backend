"""Unit tests for Gap 3 §2.2 path-edit classification (classify_path_edit)."""

from __future__ import annotations

import pytest

from abridgeai.features.career_paths.services.authoring import classify_path_edit


class TestNoEnrollmentsAlwaysSafe:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mutation": "add_course", "is_required": True},
            {"mutation": "update_stage", "min_optional_before": 0, "min_optional_after": 3},
            {"mutation": "update_stage", "enforcement_before": "soft", "enforcement_after": "hard"},
            {"mutation": "delete_stage"},
            {"mutation": "reorder_stages"},
            {"mutation": "create_stage", "new_stage_position": 1},
        ],
    )
    def test_zero_active_enrollments_is_safe(self, kwargs: dict) -> None:
        assert classify_path_edit(active_enrollments=0, **kwargs) == "safe"


class TestAddCourse:
    def test_optional_add_is_safe(self) -> None:
        assert (
            classify_path_edit(
                mutation="add_course",
                active_enrollments=3,
                stage_students_not_completed=2,
                is_required=False,
            )
            == "safe"
        )

    def test_required_add_to_unfinished_stage_is_breaking(self) -> None:
        assert (
            classify_path_edit(
                mutation="add_course",
                active_enrollments=3,
                stage_students_not_completed=2,
                is_required=True,
            )
            == "breaking"
        )

    def test_required_add_when_everyone_finished_stage_is_safe(self) -> None:
        # No one still has the stage ahead → no imposed work.
        assert (
            classify_path_edit(
                mutation="add_course",
                active_enrollments=3,
                stage_students_not_completed=0,
                is_required=True,
            )
            == "safe"
        )


class TestUpdateCourse:
    def test_flip_optional_to_required_is_breaking(self) -> None:
        assert (
            classify_path_edit(
                mutation="update_course",
                active_enrollments=1,
                stage_students_not_completed=1,
                is_required=True,
                is_required_before=False,
            )
            == "breaking"
        )

    def test_flip_required_to_optional_is_safe(self) -> None:
        assert (
            classify_path_edit(
                mutation="update_course",
                active_enrollments=1,
                stage_students_not_completed=1,
                is_required=False,
                is_required_before=True,
            )
            == "safe"
        )


class TestUpdateStage:
    def test_raise_min_optional_is_breaking(self) -> None:
        assert (
            classify_path_edit(
                mutation="update_stage",
                active_enrollments=1,
                stage_students_not_completed=1,
                min_optional_before=0,
                min_optional_after=2,
            )
            == "breaking"
        )

    def test_lower_min_optional_is_safe(self) -> None:
        assert (
            classify_path_edit(
                mutation="update_stage",
                active_enrollments=1,
                stage_students_not_completed=1,
                min_optional_before=2,
                min_optional_after=0,
            )
            == "safe"
        )

    def test_tighten_enforcement_is_breaking(self) -> None:
        assert (
            classify_path_edit(
                mutation="update_stage",
                active_enrollments=1,
                stage_students_not_completed=1,
                enforcement_before="soft",
                enforcement_after="hard",
            )
            == "breaking"
        )

    def test_loosen_enforcement_is_safe(self) -> None:
        assert (
            classify_path_edit(
                mutation="update_stage",
                active_enrollments=1,
                stage_students_not_completed=1,
                enforcement_before="hard",
                enforcement_after="soft",
            )
            == "safe"
        )

    def test_title_edit_is_safe(self) -> None:
        assert (
            classify_path_edit(mutation="update_stage", active_enrollments=1) == "safe"
        )


class TestCreateStage:
    def test_append_after_all_students_is_safe(self) -> None:
        assert (
            classify_path_edit(
                mutation="create_stage",
                active_enrollments=2,
                max_student_stage_position=2,
                new_stage_position=3,
            )
            == "safe"
        )

    def test_insert_at_or_before_student_position_is_breaking(self) -> None:
        assert (
            classify_path_edit(
                mutation="create_stage",
                active_enrollments=2,
                max_student_stage_position=2,
                new_stage_position=2,
            )
            == "breaking"
        )


class TestDeleteReorder:
    def test_delete_stage_is_breaking(self) -> None:
        assert (
            classify_path_edit(mutation="delete_stage", active_enrollments=1) == "breaking"
        )

    def test_reorder_is_breaking(self) -> None:
        assert (
            classify_path_edit(mutation="reorder_stages", active_enrollments=1)
            == "breaking"
        )
