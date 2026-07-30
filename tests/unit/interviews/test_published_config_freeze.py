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

from types import SimpleNamespace

import pytest

from abridgeai.core.exceptions import ConflictError
from abridgeai.features.interviews.schemas import InterviewConfigUpdate
from abridgeai.features.interviews.services.published_freeze import (
    PUBLISHED_EDITABLE_CONFIG_FIELDS,
    assert_config_settings_editable,
)

# Settings read by taking.py / orchestrator / evaluation.py during a run or its
# grading. Each must be refused on a published config.
FROZEN_FIELDS = [
    "persona",
    "persona_profile",
    "supported_modes",
    "tts_voice",
    "time_limit_minutes",
    "min_outcomes_to_pass",
    "practice_mode_enabled",
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
    "cooldown_hours",
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
