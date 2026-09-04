"""``security_custom_refusal_*`` applies to the FIRST refusal only.

The teacher UI offers a custom safe-refusal textarea per language, but
``safe_security_response`` only threads ``custom_refusal`` into the
REFUSE_AND_REDIRECT branch. From the second consecutive attempt
``choose_security_action`` escalates to WARN_AND_REDIRECT, whose wording is fixed
because it has to state the consequence (repeated requests are recorded for
security review) — a teacher-authored string cannot be trusted to carry that.

That is a deliberate design, not a defect, but the UI never said so: a teacher who
typed custom wording could reasonably expect it on every refusal. The scope is now
documented in the form (``security.custom_scope_hint``), and these tests pin the
behaviour the hint promises so the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
    SecurityResponsePolicy,
)
from abridgeai.features.interviews.orchestrator.security_logic import (
    choose_security_action,
    safe_security_response,
)

_CUSTOM = "Please focus on the interview question in front of you."
_QUESTION = "Explain how recursion terminates."


def _detected() -> SecurityAssessment:
    return SecurityAssessment(
        category=SecurityCategory.ANSWER_KEY_REQUEST,
        detected=True,
        confidence=0.98,
        should_block=True,
        should_record_academic_evidence=False,
        response_key="security.answer_key_request",
        normalized_fingerprint=None,
        source="rules",
    )


@pytest.mark.parametrize("language", ["en", "vi"])
def test_first_refusal_uses_the_teacher_wording(language: str) -> None:
    text = safe_security_response(
        SecurityAction.REFUSE_AND_REDIRECT,
        language=language,
        current_question=_QUESTION,
        custom_refusal=_CUSTOM,
    )
    assert _CUSTOM in text


@pytest.mark.parametrize(
    "action",
    [SecurityAction.WARN_AND_REDIRECT, SecurityAction.END_AND_FLAG],
)
def test_escalated_actions_keep_the_fixed_platform_wording(
    action: SecurityAction,
) -> None:
    """The warning must state the consequence, so it is not teacher-authored."""
    text = safe_security_response(
        action,
        language="en",
        current_question=_QUESTION,
        custom_refusal=_CUSTOM,
    )
    assert _CUSTOM not in text
    assert text.strip()


def test_warning_still_tells_the_candidate_repeats_are_recorded() -> None:
    text = safe_security_response(
        SecurityAction.WARN_AND_REDIRECT,
        language="en",
        current_question=_QUESTION,
        custom_refusal=_CUSTOM,
    )
    assert "recorded" in text.lower()


def test_custom_wording_survives_exactly_one_attempt() -> None:
    """Attempt 1 carries the teacher's text; attempt 2 onward does not.

    This is the sequence the UI hint describes, driven through the real action
    chooser rather than asserted against a hardcoded action.
    """
    shown: list[bool] = []
    for prior_attempts in range(4):
        action = choose_security_action(
            _detected(),
            consecutive_attempts=prior_attempts,
            max_consecutive_attempts=3,
            response_policy=SecurityResponsePolicy.WARN_AND_CONTINUE,
            guard_mode="enforce",
        )
        text = safe_security_response(
            action,
            language="en",
            current_question=_QUESTION,
            custom_refusal=_CUSTOM,
        )
        shown.append(_CUSTOM in text)

    assert shown == [True, False, False, False], shown


def test_continue_and_log_policy_keeps_using_the_custom_refusal() -> None:
    """That policy never escalates, so every refusal stays teacher-worded."""
    for prior_attempts in range(4):
        action = choose_security_action(
            _detected(),
            consecutive_attempts=prior_attempts,
            max_consecutive_attempts=3,
            response_policy=SecurityResponsePolicy.CONTINUE_AND_LOG,
            guard_mode="enforce",
        )
        assert action is SecurityAction.REFUSE_AND_REDIRECT
        text = safe_security_response(
            action,
            language="en",
            current_question=_QUESTION,
            custom_refusal=_CUSTOM,
        )
        assert _CUSTOM in text


def test_blank_custom_refusal_falls_back_to_the_platform_default() -> None:
    for blank in (None, "", "   "):
        text = safe_security_response(
            SecurityAction.REFUSE_AND_REDIRECT,
            language="en",
            current_question=_QUESTION,
            custom_refusal=blank,
        )
        assert "grading criteria" in text
