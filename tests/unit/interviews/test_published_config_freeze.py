"""Which interview-config settings may change after publish.

A published interview is being sat by students. Anything that alters how it is
conducted or graded must not move underneath them, or two students sit "the same"
interview under different rules. Mirrors the quiz-side freeze
(``quizzes/services/authoring.py``) on purpose.

The freeze is a WHITELIST, and the test that matters most here is
``test_a_new_field_is_frozen_by_default``: it fails when someone adds a column to
``InterviewConfigUpdate`` without deciding whether it is student-safe.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from abridgeai.core.exceptions import ConflictError
from abridgeai.features.interviews.schemas import InterviewConfigUpdate
from abridgeai.features.interviews.services.published_freeze import (
    PUBLISHED_EDITABLE_CONFIG_FIELDS,
    assert_config_settings_editable,
    assert_learning_outcomes_editable,
)

# Settings read by taking.py / orchestrator / evaluation.py during a run or its
# grading. Each must be refused on a published config.
FROZEN_FIELDS = [
    "persona",
    "persona_profile",
    "tts_voice",
    "time_limit_minutes",
    "min_outcomes_to_pass",
    "supplementary_instructions",
    "security_response_policy",
    "security_max_consecutive_attempts",
    "security_custom_refusal_en",
    "security_custom_refusal_vi",
    # Read before a session exists, so they cannot corrupt a run in flight — but
    # they are the terms of assessment. Lowering max_attempts mid-cohort strands a
    # student who already spent one; raising it gives later students more chances
    # than earlier ones got.
    "max_attempts",
    "cooldown_minutes",
    # The budgets are read into the session at dispatch; changing them mid-cohort
    # changes how hard the interviewer presses for students who start later.
    "max_follow_ups_per_question",
    "max_hints_per_question",
]


def _config(status: str) -> SimpleNamespace:
    """Stand-in for an InterviewConfig; the guard only reads ``status``."""
    return SimpleNamespace(status=status)


def _assert_editable(status: str, changed: set[str]) -> None:
    assert_config_settings_editable(_config(status), changed)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", FROZEN_FIELDS)
def test_conduct_and_grading_settings_are_frozen_once_published(field: str) -> None:
    with pytest.raises(ConflictError) as exc:
        _assert_editable("published", {field})
    message = str(exc.value)
    assert "interview_published_setting_locked" in message
    # The client has to be able to point at the offending field.
    assert field in message


@pytest.mark.parametrize("field", sorted(PUBLISHED_EDITABLE_CONFIG_FIELDS))
def test_student_safe_settings_stay_editable_after_publish(field: str) -> None:
    _assert_editable("published", {field})


@pytest.mark.parametrize("status", ["draft", "archived"])
@pytest.mark.parametrize("field", ["persona", "time_limit_minutes", "min_outcomes_to_pass"])
def test_unpublished_configs_are_unrestricted(status: str, field: str) -> None:
    """Unpublishing is the documented escape hatch, so it must actually work."""
    _assert_editable(status, {field})


def test_a_mixed_patch_is_rejected_and_names_only_the_frozen_fields() -> None:
    with pytest.raises(ConflictError) as exc:
        _assert_editable("published", {"title", "persona", "time_limit_minutes"})
    message = str(exc.value)
    assert "persona" in message
    assert "time_limit_minutes" in message
    # ``title`` is allowed, so blaming it would send the teacher to the wrong field.
    assert "title" not in message.split(":")[-1].split(".")[0]


def test_an_empty_patch_is_allowed() -> None:
    """A no-op PATCH must not 409 — the UI may submit an unchanged form."""
    _assert_editable("published", set())


def test_a_new_field_is_frozen_by_default() -> None:
    """Guard against a future column silently becoming editable after publish.

    If you are here because this test failed, you added a field to
    ``InterviewConfigUpdate``. Decide deliberately:

    * it cannot affect a live or graded attempt -> add it to
      ``PUBLISHED_EDITABLE_CONFIG_FIELDS`` and to the allow-list below;
    * otherwise -> add it to ``FROZEN_FIELDS`` above so the freeze is asserted.

    Do NOT just widen the whitelist to make this pass.
    """
    known = set(FROZEN_FIELDS) | PUBLISHED_EDITABLE_CONFIG_FIELDS
    patchable = set(InterviewConfigUpdate.model_fields.keys())
    unclassified = patchable - known
    assert not unclassified, (
        f"unclassified InterviewConfigUpdate field(s): {sorted(unclassified)} — "
        "decide whether each is student-safe after publish (see this test's docstring)"
    )


def test_the_whitelist_only_contains_real_patchable_fields() -> None:
    """A typo in the whitelist would silently freeze a field meant to be editable."""
    patchable = set(InterviewConfigUpdate.model_fields.keys())
    assert patchable >= PUBLISHED_EDITABLE_CONFIG_FIELDS


class TestLearningOutcomesFreeze:
    """Outcomes are the grading criteria, so they freeze with the rest.

    The settings freeze above covers the PATCHed config fields. Outcomes are a
    separate collection (create/update/delete endpoints), so they get their own
    guard — but the same policy, the same error code, and the same escape hatch
    (unpublish).
    """

    @pytest.mark.parametrize(
        "status",
        ["draft", "archived"],
    )
    def test_unpublished_configs_may_change_outcomes(self, status: str) -> None:
        # Unpublishing is the documented escape hatch; it must actually work.
        assert_learning_outcomes_editable(_config(status))  # type: ignore[arg-type]

    def test_a_published_config_rejects_outcome_changes(self) -> None:
        with pytest.raises(ConflictError) as exc:
            assert_learning_outcomes_editable(_config("published"))  # type: ignore[arg-type]
        message = str(exc.value)
        assert "interview_published_setting_locked" in message
        # The teacher has to know WHICH thing is frozen and how to unfreeze it.
        assert "learning outcomes" in message
        assert "Unpublish" in message

    def test_the_guard_fires_before_any_write(self) -> None:
        """Prove the service functions gate on the guard before touching the db.

        Each outcome mutation must fetch the config and refuse a published one
        without creating/updating/deleting anything — otherwise the UI dimming
        would be the only line of defence.
        """
        from unittest.mock import AsyncMock, patch

        from abridgeai.features.interviews.services import authoring as svc  # noqa: PLC0415

        published = _config("published")
        db = AsyncMock()
        config_id = "00000000-0000-0000-0000-000000000001"
        outcome_id = "00000000-0000-0000-0000-000000000002"

        with patch.object(svc, "_require_config", new=AsyncMock(return_value=published)):
            # add_outcome: refuses before computing a position or inserting.
            with (
                patch.object(svc.authoring_queries, "next_outcome_position", new=AsyncMock()) as pos,
                patch.object(svc, "flush_or_conflict", new=AsyncMock()) as flush,
            ):
                with pytest.raises(ConflictError):
                    asyncio.run(
                        svc.add_outcome(  # type: ignore[arg-type]
                            db,
                            config_id,
                            SimpleNamespace(
                                model_dump=lambda exclude_unset: {
                                    "outcome_text": "x",
                                    "outcome_type": "knowledge",
                                }
                            ),
                            object(),
                        )
                    )
                pos.assert_not_awaited()
                flush.assert_not_awaited()

            # update_outcome: refuses before touching the existing row.
            with (
                patch.object(svc, "_require_outcome", new=AsyncMock()) as require_outcome,
                patch.object(svc, "_apply_patch") as apply_patch,
                patch.object(svc, "flush_or_conflict", new=AsyncMock()) as flush,
            ):
                with pytest.raises(ConflictError):
                    asyncio.run(
                        svc.update_outcome(  # type: ignore[arg-type]
                            db, config_id, outcome_id, SimpleNamespace(), object()
                        )
                    )
                require_outcome.assert_not_awaited()
                apply_patch.assert_not_called()
                flush.assert_not_awaited()

            # delete_outcome: refuses before loading or soft-deleting the row.
            with (
                patch.object(svc, "_require_outcome", new=AsyncMock()) as require_outcome,
                patch.object(svc, "soft_delete_cascade", new=AsyncMock()) as soft_delete,
            ):
                with pytest.raises(ConflictError):
                    asyncio.run(
                        svc.delete_outcome(  # type: ignore[arg-type]
                            db, config_id, outcome_id, object()
                        )
                    )
                require_outcome.assert_not_awaited()
                soft_delete.assert_not_awaited()
