"""Browser-integrity policy for quiz attempts (migration 0115).

The SCORING RULES are not duplicated here. ``interviews.schemas.integrity``
already owns them — weights, clamping, threshold arithmetic, the retry-key
parser — and they are about assessment integrity in general, not about
interviews: the events live in one shared table (``assessment_integrity_events``,
discriminated by ``assessment_kind``) and a tab switch must not be worth a
different number of points depending on which assessment a student is sitting.
Re-implementing them for quizzes would be two copies of one policy, free to
drift apart silently.

What IS quiz-specific is where the policy comes from — a ``Quiz`` row rather
than an ``InterviewConfig`` — and that is all this module provides.
"""

from __future__ import annotations

from typing import Any

# Shipped defaults, duplicated from the shared contract only as a fallback for
# a row that somehow predates the columns. Kept in one tuple so the two uses
# below cannot disagree.
_DEFAULTS = {
    "tab_switch": 3,
    "focus_lost": 1,
    "fullscreen_exit": 2,
    "score_threshold": 3,
}


def integrity_policy_snapshot_from_quiz(quiz: Any) -> dict[str, int]:  # noqa: ANN401 -- ORM row
    """The integrity policy a NEW attempt is scored under.

    Frozen onto the attempt row at start, so a quiz edited — or its weights
    retuned — mid-cohort never re-scores an attempt that began under the
    earlier rules. Keys are the canonical spellings from
    ``interviews.schemas.integrity.SNAPSHOT_KEYS``, because the scoring
    functions look them up by those names.

    ``getattr`` with a default rather than direct attribute access: this is
    also called for attempts whose snapshot is empty (rows created before the
    columns existed), where ``quiz`` may be ``None``.
    """
    return {
        "tab_switch": int(
            getattr(quiz, "integrity_weight_tab_switch", _DEFAULTS["tab_switch"])
            or _DEFAULTS["tab_switch"]
        ),
        "focus_lost": int(
            getattr(quiz, "integrity_weight_focus_lost", _DEFAULTS["focus_lost"])
            or _DEFAULTS["focus_lost"]
        ),
        "fullscreen_exit": int(
            getattr(quiz, "integrity_weight_fullscreen_exit", _DEFAULTS["fullscreen_exit"])
            or _DEFAULTS["fullscreen_exit"]
        ),
        "score_threshold": int(
            getattr(quiz, "integrity_score_threshold", _DEFAULTS["score_threshold"])
            or _DEFAULTS["score_threshold"]
        ),
    }


__all__ = ["integrity_policy_snapshot_from_quiz"]
