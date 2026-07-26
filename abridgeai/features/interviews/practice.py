"""Practice-mode constants and the bank-partition rule.

A practice run is a rehearsal: it does not consume an attempt, anchors no
cooldown, is never evaluated, and therefore never writes ``pass_verdict`` — so
it can never satisfy the SR-lesson or quiz unlock gates that read that column.

The one thing that is easy to get wrong, and the reason this module exists
rather than the rule being inlined at each call site, is the bank partition.
``practice_only`` is a boolean on every question, and every query that picks a
question for a RUNNING session must filter on it. Miss one and a rehearsal
reveals the real exam; miss the one inside the evaluator and unasked practice
questions are counted as unanswered against a real grade.

Deliberately NOT filtered: ``protected_content_for_config`` loads the whole bank
regardless of partition, which is what stops the interviewer uttering questions
from the other side of the split. Filtering it would break that property, not
strengthen it.
"""

from __future__ import annotations

from typing import Literal, get_args

SessionMode = Literal["assessment", "practice"]

MODE_ASSESSMENT: SessionMode = "assessment"
MODE_PRACTICE: SessionMode = "practice"

SESSION_MODES: tuple[SessionMode, ...] = get_args(SessionMode)

PRACTICE_MAX_RUNS = 1
"""How many practice runs a student gets per config.

One, not unlimited: practice draws from a partition of the same bank, so
repeated runs would still farm those questions. It is a rehearsal of the
interface, not a study aid.
"""

PRACTICE_ATTEMPT_NUMBER = 0
"""``attempt_number`` for every practice session.

Real attempts occupy 1..n (``get_session_attempt_number`` allocates
``MAX(...) + 1``), so 0 sits outside that space and cannot collide under
``uq_interview_sessions_number``. Verified safe: nothing in the feature compares
``attempt_number`` — every use is a projection into a DTO. It IS rendered,
though, so surfaces showing a practice session must print the mode instead of
"attempt 0".
"""


def is_practice(mode: str | None) -> bool:
    """True only for an explicit practice run.

    ``None`` and unknown values read as assessment. That direction is chosen on
    purpose: a session whose mode we cannot determine gets graded, which is
    recoverable, rather than silently discarded.
    """
    return mode == MODE_PRACTICE


def partition_for_mode(mode: str | None) -> bool:
    """The ``practice_only`` value a session in ``mode`` is allowed to draw.

    Pass the result to ``list_questions_for_config(..., practice_only=...)``.
    """
    return is_practice(mode)
