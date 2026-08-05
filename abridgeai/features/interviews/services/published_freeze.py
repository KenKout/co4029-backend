"""Which interview-config settings may change after the config is published.

A published interview is being sat by students. Anything that alters how it is
conducted, graded, or retaken must not move underneath them, or two students sit
"the same" interview under different rules.

Mirrors the quiz-side freeze (``quizzes/services/authoring.py``) deliberately:
same reasoning, same shape, so the two features behave alike. Lives in its own
module rather than inside ``services/authoring.py`` because it is a self-contained
policy — and because the LOC gate in ``tests/integration/test_interviews_metric.py``
caps feature files at 800 lines, which authoring.py was pushed past when this
started life there.
"""

from __future__ import annotations

from abridgeai.core.exceptions import ConflictError
from abridgeai.features.interviews.models import InterviewConfig

# A WHITELIST, not a blacklist — any column added later is frozen by default
# until someone explicitly vets it. That is the whole point; forgetting to freeze
# a new scoring knob is far more damaging than forgetting to unfreeze a new
# cosmetic one. ``tests/unit/interviews/test_published_config_freeze.py`` fails
# when a new ``InterviewConfigUpdate`` field is neither frozen nor vetted here.
#
# Why each of these is safe:
#   title                              teacher-facing label; never enters the
#                                      interview prompt or the transcript
#   security_incident_summary_enabled  controls a teacher-side report only
#                                      (routers/authoring.py), never the run
#   lock_quiz_ef_until_pass            downstream SR/quiz gating, read outside
#                                      the interview itself
#
# ``max_attempts`` and ``cooldown_hours`` are deliberately NOT here, even though
# they are only read before a session exists and so cannot disturb a run in
# flight. They are still the terms of assessment: lowering ``max_attempts``
# mid-cohort can retroactively strand a student who already used an attempt in
# good faith, and raising it hands later students more chances than earlier ones
# got. "Cannot corrupt a live run" is a weaker test than "is fair to everyone
# sitting the same published interview", and the second is the one that matters.
#
# Everything else is frozen: time_limit_minutes, min_outcomes_to_pass, persona,
# persona_profile, supported_modes, tts_voice, supplementary_instructions,
# practice_mode_enabled and the security response knobs are all read by
# ``services/taking.py`` / ``orchestrator/`` while an interview runs, or by
# ``services/evaluation.py`` when it is graded.
PUBLISHED_EDITABLE_CONFIG_FIELDS = frozenset(
    {
        "title",
        "security_incident_summary_enabled",
        "lock_quiz_ef_until_pass",
    }
)


def assert_config_settings_editable(config: InterviewConfig, changed_fields: set[str]) -> None:
    """Field-aware freeze for the config PATCH on a published interview.

    Draft and archived configs are unrestricted. On a published one, only
    :data:`PUBLISHED_EDITABLE_CONFIG_FIELDS` may change; anything that would
    alter how the interview is conducted, graded, or retaken is rejected, naming
    the offending fields so the client can point at them.
    """
    if config.status != "published":
        return
    frozen = changed_fields - PUBLISHED_EDITABLE_CONFIG_FIELDS
    if frozen:
        raise ConflictError(
            "interview_published_setting_locked: these settings are frozen on a "
            "published interview because they change how it is conducted, graded, "
            f"or retaken: {', '.join(sorted(frozen))}. Unpublish the interview "
            "first to change them."
        )


def assert_learning_outcomes_editable(config: InterviewConfig) -> None:
    """Freeze learning-outcome mutations on a published interview.

    The outcomes ARE the grading criteria: evaluation.py compares each answer
    against them and weights the result by ``importance_weight``. Adding,
    removing, or reweighting an outcome mid-cohort means two students sit \"the
    same\" interview judged against different standards — strictly worse than a
    frozen settings field, because it silently changes how already-submitted
    answers would score. The config PATCH freeze covers the pass threshold
    (``min_outcomes_to_pass``); this covers the criteria themselves.

    Called by the outcome create/update/delete service functions before any
    write, so a published config rejects the mutation with the same
    ``interview_published_setting_locked`` error the settings PATCH uses — the
    client treats both identically. Unpublishing lifts the restriction.
    """
    if config.status != "published":
        return
    raise ConflictError(
        "interview_published_setting_locked: learning outcomes are frozen on a "
        "published interview because the AI judges answers against them. "
        "Unpublish the interview first to change them."
    )


__all__ = [
    "PUBLISHED_EDITABLE_CONFIG_FIELDS",
    "assert_config_settings_editable",
    "assert_learning_outcomes_editable",
]
