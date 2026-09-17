"""The teacher-selectable response to a quiz proctoring threshold crossing.

The policy governs DISCLOSURE, never judgement: under ``continue_and_log`` the
score still accrues, the evidence row is still written and the attempt is
still flagged for the teacher — only the learner-facing warning is withheld.
Most of what is worth asserting here is therefore about what the policy does
NOT change, which is why the scoring assertions outnumber the warning ones.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import String

from abridgeai.core.exceptions import AppError
from abridgeai.features.quizzes.models import Quiz
from abridgeai.features.quizzes.schemas.authoring import QuizAuthoring
from abridgeai.features.quizzes.services.authoring import _coerce_patch_value
from abridgeai.features.quizzes.services.integrity import (
    DEFAULT_INTEGRITY_RESPONSE_POLICY,
    INTEGRITY_RESPONSE_POLICIES,
    integrity_policy_snapshot_from_quiz,
    warns_the_learner,
)

WARN = "warn_and_continue"
SILENT = "continue_and_log"


def _quiz(policy: str | None = None) -> SimpleNamespace:
    """A quiz row stub carrying the integrity columns the snapshot reads."""
    fields = {
        "integrity_weight_tab_switch": 3,
        "integrity_weight_focus_lost": 1,
        "integrity_weight_fullscreen_exit": 2,
        "integrity_score_threshold": 3,
        "require_camera": False,
    }
    if policy is not None:
        fields["integrity_response_policy"] = policy
    return SimpleNamespace(**fields)


# --------------------------------------------------------------------------- #
# Column shape — the default is what keeps the migration inert on existing rows
# --------------------------------------------------------------------------- #


def test_policy_column_defaults_to_warning_so_existing_quizzes_are_unchanged() -> None:
    column = Quiz.__table__.c.integrity_response_policy
    assert isinstance(column.type, String)
    assert column.nullable is False
    assert WARN in str(column.server_default.arg)


def test_policy_column_is_check_constrained_to_the_two_known_values() -> None:
    constraint = next(
        c
        for c in Quiz.__table__.constraints
        if getattr(c, "name", None) == "ck_quizzes_integrity_response_policy"
    )
    expression = str(constraint.sqltext)
    for value in INTEGRITY_RESPONSE_POLICIES:
        assert value in expression


# --------------------------------------------------------------------------- #
# The snapshot — an attempt is judged under the policy frozen at its start
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("policy", [WARN, SILENT])
def test_snapshot_freezes_the_policy_in_force_when_the_attempt_began(policy: str) -> None:
    assert integrity_policy_snapshot_from_quiz(_quiz(policy))["response_policy"] == policy


def test_snapshot_falls_back_to_warning_for_a_quiz_row_predating_the_column() -> None:
    # A row loaded before migration 0128 has no attribute at all; it ran under
    # the hardcoded warn, so that is what it must keep being judged by.
    assert integrity_policy_snapshot_from_quiz(_quiz())["response_policy"] == WARN


def test_snapshot_falls_back_to_warning_for_a_missing_quiz() -> None:
    # The snapshot builder is also called for attempts whose own snapshot is
    # empty, where the quiz may be None entirely.
    assert integrity_policy_snapshot_from_quiz(None)["response_policy"] == WARN


def test_policy_does_not_disturb_the_weights_or_threshold_it_travels_with() -> None:
    warned = integrity_policy_snapshot_from_quiz(_quiz(WARN))
    silent = integrity_policy_snapshot_from_quiz(_quiz(SILENT))
    scoring_keys = ("tab_switch", "focus_lost", "fullscreen_exit", "score_threshold")
    assert {k: warned[k] for k in scoring_keys} == {k: silent[k] for k in scoring_keys}


# --------------------------------------------------------------------------- #
# The decision — read out of the frozen snapshot, never the live quiz row
# --------------------------------------------------------------------------- #


def test_warning_is_withheld_only_under_the_silent_policy() -> None:
    assert warns_the_learner({"response_policy": SILENT}) is False
    assert warns_the_learner({"response_policy": WARN}) is True


@pytest.mark.parametrize("snapshot", [None, {}, {"score_threshold": 3}])
def test_a_snapshot_without_a_policy_warns(snapshot: dict[str, object] | None) -> None:
    # Every attempt started before 0128 has such a snapshot. Defaulting to
    # silence would retroactively stop warning learners mid-cohort.
    assert warns_the_learner(snapshot) is True


@pytest.mark.parametrize("junk", ["", "  ", "nonsense", None, 0, False])
def test_an_unrecognised_policy_value_warns_rather_than_silently_disabling(
    junk: object,
) -> None:
    # Direction-safe: a corrupt snapshot must degrade towards telling the
    # learner, not towards a quiz that silently stopped warning anyone.
    assert warns_the_learner({"response_policy": junk}) is True


# --------------------------------------------------------------------------- #
# The authoring boundary — a bad value is a named 400, not a database error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("policy", list(INTEGRITY_RESPONSE_POLICIES))
def test_patch_accepts_each_known_policy(policy: str) -> None:
    assert _coerce_patch_value("integrity_response_policy", policy) == policy


@pytest.mark.parametrize("bad", ["end_and_flag", "warn", "", None, 1, True])
def test_patch_rejects_an_unknown_policy_naming_the_field(bad: object) -> None:
    with pytest.raises(AppError) as excinfo:
        _coerce_patch_value("integrity_response_policy", bad)
    message = str(excinfo.value)
    assert "integrity_response_policy" in message
    # The message lists what IS allowed — a teacher hitting this needs to know
    # the two valid values, not merely that theirs was wrong.
    for value in INTEGRITY_RESPONSE_POLICIES:
        assert value in message


def test_end_and_flag_is_not_borrowed_from_the_interview_security_vocabulary() -> None:
    # ``security_response_policy`` on interview_configs permits a third value.
    # That policy governs the AI guard's reaction to prompt injection, not
    # proctoring, and there is no proctoring behaviour behind it here.
    assert "end_and_flag" not in INTEGRITY_RESPONSE_POLICIES


# --------------------------------------------------------------------------- #
# Exposure — teacher-only, for the same reason the weights are
# --------------------------------------------------------------------------- #


def test_policy_is_exposed_on_the_authoring_projection() -> None:
    assert "integrity_response_policy" in QuizAuthoring.model_fields
    assert (
        QuizAuthoring.model_fields["integrity_response_policy"].default
        == DEFAULT_INTEGRITY_RESPONSE_POLICY
    )


def test_policy_is_not_exposed_to_learners() -> None:
    # Telling a student the crossing will be silent tells them they will not
    # be caught, which defeats the setting entirely.
    from abridgeai.features.quizzes.schemas.public import QuizPublic

    assert "integrity_response_policy" not in QuizPublic.model_fields
