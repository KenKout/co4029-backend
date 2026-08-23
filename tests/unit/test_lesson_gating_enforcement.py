"""Unit tests for FR-4.5 / FR-5.3 gating enforcement.

Covers the two new enforcement points added in phase-02:

* ``courses.routers.learner._ensure_lesson_unlocked`` — 403 with the
  unlock-requirements payload when ``check_lesson_unlock`` says locked.
Both helpers consult ``settings.lesson_gating_enforced`` (emergency
off-switch) and the spaced_repetition public API; everything external is
mocked — no DB.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from abridgeai.features.courses.routers.learner import _ensure_lesson_unlocked
from abridgeai.features.spaced_repetition.api.public import LessonUnlockStatus

_STUDENT = uuid.uuid4()
_LESSON = uuid.uuid4()
_QUIZ = uuid.uuid4()
_MODULE = uuid.uuid4()
_CONFIG = uuid.uuid4()


def _unlock_status(*, eligible: bool) -> LessonUnlockStatus:
    return LessonUnlockStatus(
        eligible=eligible,
        current_ratio=0.5,
        required_ratio=0.8,
        ef_min=2.0,
        total_cards=10,
        passing_cards=5,
        blocking_cards=[],
        prereq_lesson_ids_unlocked=True,
        interview_pass_required=False,
        interview_passed=False,
    )


def _settings(*, enforced: bool) -> SimpleNamespace:
    return SimpleNamespace(lesson_gating_enforced=enforced)


def _fake_user() -> SimpleNamespace:
    return SimpleNamespace(user_id=_STUDENT)


class TestEnsureLessonUnlocked:
    async def test_noop_when_flag_disabled(self) -> None:
        with (
            patch(
                "abridgeai.features.courses.routers.learner.get_settings",
                return_value=_settings(enforced=False),
            ),
            patch(
                "abridgeai.features.spaced_repetition.api.public.check_lesson_unlock",
                new=AsyncMock(side_effect=AssertionError("must not be called")),
            ),
        ):
            await _ensure_lesson_unlocked(AsyncMock(), _fake_user(), _LESSON)

    async def test_noop_when_eligible(self) -> None:
        with (
            patch(
                "abridgeai.features.courses.routers.learner.get_settings",
                return_value=_settings(enforced=True),
            ),
            patch(
                "abridgeai.features.spaced_repetition.api.public.check_lesson_unlock",
                new=AsyncMock(return_value=_unlock_status(eligible=True)),
            ),
        ):
            await _ensure_lesson_unlocked(AsyncMock(), _fake_user(), _LESSON)

    async def test_403_with_requirements_when_locked(self) -> None:
        with (
            patch(
                "abridgeai.features.courses.routers.learner.get_settings",
                return_value=_settings(enforced=True),
            ),
            patch(
                "abridgeai.features.spaced_repetition.api.public.check_lesson_unlock",
                new=AsyncMock(return_value=_unlock_status(eligible=False)),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _ensure_lesson_unlocked(AsyncMock(), _fake_user(), _LESSON)

        exc = exc_info.value
        assert exc.status_code == 403
        assert exc.detail["error"] == "lesson_locked"
        assert exc.detail["lesson_id"] == str(_LESSON)
        assert exc.detail["required_ratio"] == 0.8
        assert exc.detail["passing_cards"] == 5
        assert exc.detail["prerequisites_met"] is True



class TestGateWiring:
    """Call-site coverage: deleting any `await _ensure_*` line must fail here."""

    async def test_get_lesson_route_invokes_gate(self) -> None:
        from abridgeai.features.courses.routers import learner as courses_learner

        gate = AsyncMock()
        with (
            patch.object(
                courses_learner.catalog_service,
                "get_published_lesson_for_learner",
                new=AsyncMock(return_value=SimpleNamespace(id=_LESSON)),
            ),
            patch.object(courses_learner, "_ensure_lesson_unlocked", new=gate),
        ):
            await courses_learner.get_lesson(_LESSON, _fake_user(), AsyncMock())
        gate.assert_awaited_once()
        assert gate.await_args.args[2] == _LESSON

    async def test_lesson_resources_route_invokes_gate(self) -> None:
        from abridgeai.features.courses.routers import learner as courses_learner

        gate = AsyncMock()
        with (
            patch.object(
                courses_learner.catalog_service,
                "list_visible_lesson_resources_for_learner",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(courses_learner, "_ensure_lesson_unlocked", new=gate),
        ):
            await courses_learner.list_lesson_resources(_LESSON, _fake_user(), AsyncMock())
        gate.assert_awaited_once()

    async def test_download_url_route_invokes_gate(self) -> None:
        from abridgeai.features.courses.routers import learner as courses_learner

        gate = AsyncMock()
        resource_id = uuid.uuid4()
        download = SimpleNamespace(
            url="https://example.test/u", expires_at=SimpleNamespace(isoformat=lambda: "t")
        )
        with (
            patch.object(
                courses_learner.catalog_service,
                "get_visible_resource_lesson_id",
                new=AsyncMock(return_value=_LESSON),
            ),
            patch.object(
                courses_learner.catalog_service,
                "get_lesson_resource_download_url",
                new=AsyncMock(return_value=download),
            ),
            patch.object(courses_learner, "_ensure_lesson_unlocked", new=gate),
        ):
            await courses_learner.get_lesson_resource_download_url(
                resource_id, _fake_user(), AsyncMock()
            )
        gate.assert_awaited_once()
        assert gate.await_args.args[2] == _LESSON

    async def test_material_routes_invoke_gate(self) -> None:
        from abridgeai.features.materials.routers import learner as materials_learner

        gate = AsyncMock()
        material_id = uuid.uuid4()
        material = SimpleNamespace(lesson_id=_LESSON)
        with (
            patch.object(
                materials_learner.catalog_service,
                "get_visible_material_for_user",
                new=AsyncMock(return_value=material),
            ),
            patch.object(
                materials_learner.catalog_service,
                "get_stream_url_for_material",
                new=AsyncMock(return_value=SimpleNamespace()),
            ),
            patch.object(
                materials_learner.catalog_service,
                "list_visible_chunks_preview",
                new=AsyncMock(return_value=[]),
            ),
            patch.object(materials_learner, "_ensure_owning_lesson_unlocked", new=gate),
        ):
            await materials_learner.get_material(material_id, _fake_user(), AsyncMock())
            await materials_learner.get_material_stream_url(material_id, _fake_user(), AsyncMock())
            await materials_learner.get_material_chunks_preview(
                material_id, _fake_user(), AsyncMock(), 5
            )
        assert gate.await_count == 3
        assert all(call.args[2] == _LESSON for call in gate.await_args_list)


class TestSlimLessonProjection:
    """List/tree payloads must not carry the lesson body past the gate."""

    def test_slim_lesson_strips_body_fields(self) -> None:
        from abridgeai.features.courses.services.catalog import _slim_lesson_public

        lesson = SimpleNamespace(
            id=_LESSON,
            title="T",
            lesson_type="reading",
            summary="spoiler",
            notes_markdown="# full body",
            primary_material_id=uuid.uuid4(),
            estimated_minutes=5,
            difficulty="easy",
        )
        public = _slim_lesson_public(lesson)
        assert public.summary is None
        assert public.notes_markdown is None
        assert public.primary_material_id is None
        assert public.title == "T"
        assert public.estimated_minutes == 5
