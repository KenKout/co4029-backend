"""Question-type-aware grader.

Centralises the answer-grading logic so the ``answer_attempt`` service
stays small. Each question type has its own grader function returning a
``GradeResult`` (``is_correct``, ``points_awarded``).

Conventions
-----------
* MCQ + true_false: grade by looking up the selected option's
  ``is_correct`` flag (the canonical answer). The legacy contract used
  one ``selected_option_id`` per answer; we honour that and ignore the
  multi-select list shape until the platform actually adds multi-select
  questions.
* short_answer: case-insensitive whitespace-normalised exact match
  against ``original_generated_payload['correct_answer']``. Treats
  hyphenated and unhyphenated variants as equivalent
  ("time-variant" == "time variant").
* fill_blank: positional list compare. Student's drag-drop slots are
  serialised into ``answer_text`` as a JSON array; mismatches in length
  fail outright.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from abridgeai.features.quizzes.models import QuizQuestion, QuizQuestionOption

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class GradeResult:
    is_correct: bool
    points_awarded: Decimal


_ZERO = GradeResult(is_correct=False, points_awarded=Decimal("0"))
_ONE = GradeResult(is_correct=True, points_awarded=Decimal("1"))


async def grade_answer(
    db: AsyncSession,
    *,
    question_id: UUID,
    selected_option_id: UUID | None,
    answer_text: str | None,
) -> GradeResult:
    """Grade one answer against its question's canonical answer."""
    question = await db.get(QuizQuestion, question_id)
    if question is None:
        return _ZERO

    qtype = question.question_type
    if qtype in {"multiple_choice", "true_false"}:
        # Phase 7: multi-select MCQ (single_answer=False) grades all-correct-and-
        # no-incorrect against the JSON list of chosen option ids in answer_text;
        # single-select keeps the legacy single selected_option_id path.
        if not getattr(question, "single_answer", True):
            return await _grade_multi_select(db, question.id, answer_text)
        return await _grade_by_option(db, question.id, selected_option_id)
    if qtype == "short_answer":
        return _grade_short_answer(question, answer_text)
    if qtype == "fill_blank":
        return _grade_fill_blank(question, answer_text)
    if qtype == "numerical":
        return _grade_numerical(question, answer_text)
    if qtype == "matching":
        return _grade_matching(question, answer_text)
    if qtype == "ordering":
        return _grade_ordering(question, answer_text)
    return _ZERO


async def _grade_multi_select(
    db: AsyncSession, question_id: UUID, answer_text: str | None
) -> GradeResult:
    """All-or-nothing multi-select: chosen set must equal the correct set."""
    from sqlalchemy import select  # noqa: PLC0415

    chosen = _parse_id_list(answer_text)
    rows = (
        await db.execute(
            select(QuizQuestionOption.id, QuizQuestionOption.is_correct).where(
                QuizQuestionOption.question_id == question_id
            )
        )
    ).all()
    if not rows:
        return _ZERO
    correct_ids = {str(oid) for oid, is_correct in rows if is_correct}
    chosen_ids = {str(c) for c in chosen}
    return _ONE if chosen_ids and chosen_ids == correct_ids else _ZERO


def _grade_numerical(question: QuizQuestion, answer_text: str | None) -> GradeResult:
    """Numeric answer within an accepted tolerance (|submitted - expected| <= tol)."""
    expected = getattr(question, "numeric_answer", None)
    if expected is None or not answer_text:
        return _ZERO
    try:
        submitted = Decimal(str(answer_text).strip())
        expected_value = Decimal(str(expected))
        tolerance = Decimal(str(getattr(question, "numeric_tolerance", None) or 0))
    except (TypeError, ValueError, ArithmeticError):
        return _ZERO
    return _ONE if abs(submitted - expected_value) <= tolerance else _ZERO


def _grade_matching(question: QuizQuestion, answer_text: str | None) -> GradeResult:
    """All pairs must match. match_pairs = [{"left":..,"right":..}]; submission is
    a JSON object {left: chosen_right}."""
    pairs = getattr(question, "match_pairs", None)
    if not isinstance(pairs, list) or not pairs or not answer_text:
        return _ZERO
    try:
        submitted = json.loads(answer_text)
    except (TypeError, ValueError):
        return _ZERO
    if not isinstance(submitted, dict):
        return _ZERO
    for pair in pairs:
        left = str(pair.get("left"))
        if _normalize_text(str(submitted.get(left, ""))) != _normalize_text(
            str(pair.get("right", ""))
        ):
            return _ZERO
    return _ONE


def _grade_ordering(question: QuizQuestion, answer_text: str | None) -> GradeResult:
    """Exact-sequence match. ordering_sequence is the correct ordered list; the
    submission is a JSON array in the student's chosen order."""
    expected = getattr(question, "ordering_sequence", None)
    if not isinstance(expected, list) or not expected or not answer_text:
        return _ZERO
    try:
        submitted = json.loads(answer_text)
    except (TypeError, ValueError):
        return _ZERO
    if not isinstance(submitted, list) or len(submitted) != len(expected):
        return _ZERO
    return (
        _ONE
        if all(str(s) == str(e) for s, e in zip(submitted, expected, strict=True))
        else _ZERO
    )


def _parse_id_list(answer_text: str | None) -> list[str]:
    """Decode a JSON array of option ids (multi-select submission)."""
    if not answer_text:
        return []
    try:
        parsed = json.loads(answer_text)
    except (TypeError, ValueError):
        return [p.strip() for p in answer_text.split(",") if p.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


async def _grade_by_option(
    db: AsyncSession, question_id: UUID, selected_option_id: UUID | None
) -> GradeResult:
    if selected_option_id is None:
        return _ZERO
    from sqlalchemy import select  # noqa: PLC0415

    option = (
        await db.execute(
            select(QuizQuestionOption).where(
                QuizQuestionOption.id == selected_option_id,
                QuizQuestionOption.question_id == question_id,
            )
        )
    ).scalar_one_or_none()
    if option is None:
        return _ZERO
    return _ONE if option.is_correct else _ZERO


def _grade_short_answer(question: QuizQuestion, answer_text: str | None) -> GradeResult:
    expected = _expected_short_answer(question)
    if not expected or not answer_text:
        return _ZERO
    return _ONE if _normalize_text(answer_text) == _normalize_text(expected) else _ZERO


def _grade_fill_blank(question: QuizQuestion, answer_text: str | None) -> GradeResult:
    expected = _expected_fill_blank(question)
    submitted = _parse_fill_blank_submission(answer_text)
    if not expected or len(expected) != len(submitted):
        return _ZERO
    for sub, exp in zip(submitted, expected, strict=True):
        if _normalize_text(sub) != _normalize_text(exp):
            return _ZERO
    return _ONE


def _expected_short_answer(question: QuizQuestion) -> str | None:
    payload = question.original_generated_payload or {}
    expected = payload.get("correct_answer")
    return expected if isinstance(expected, str) else None


def _expected_fill_blank(question: QuizQuestion) -> list[str]:
    payload = question.original_generated_payload or {}
    blanks = payload.get("correct_answer")
    if isinstance(blanks, list):
        return [str(b) for b in blanks if isinstance(b, str)]
    return []


def _parse_fill_blank_submission(answer_text: str | None) -> list[str]:
    """Decode the student's drag-drop submission.

    The frontend serialises slot values as a JSON array (one entry per
    blank). We accept a comma-separated string as a fallback so manual
    API users / curl smoke tests still work.
    """
    if not answer_text:
        return []
    try:
        parsed = json.loads(answer_text)
    except (TypeError, ValueError):
        return [piece.strip() for piece in answer_text.split(",") if piece.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("-", " ").lower().split())


def grade_answer_against_revision(
    revision_payload: dict,
    *,
    selected_option_key: str | None,
    answer_text: str | None,
    selected_option_keys: list[str] | None = None,
) -> GradeResult:
    """Grade an answer against a question-definition *snapshot* dict.

    Unlike :func:`grade_answer` (which reads live DB rows), this grades against
    a plain dict shaped like a ``QuizQuestionRevision.payload_json`` / live
    snapshot — so a regrade can judge each stored answer against the CURRENT
    question definition (Phase 1). MCQ/true_false grade by ``option_key`` (stable
    across edits), not option UUID.
    """
    qtype = revision_payload.get("question_type")
    if qtype in {"multiple_choice", "true_false"}:
        options = revision_payload.get("options", [])
        if not revision_payload.get("single_answer", True):
            correct_keys = {
                str(opt.get("option_key"))
                for opt in options
                if isinstance(opt, dict) and opt.get("is_correct")
            }
            chosen_keys = {str(key) for key in selected_option_keys or []}
            return _ONE if chosen_keys and chosen_keys == correct_keys else _ZERO
        if selected_option_key is None:
            return _ZERO
        for opt in options:
            if opt.get("option_key") == selected_option_key:
                return _ONE if opt.get("is_correct") else _ZERO
        return _ZERO
    if qtype == "short_answer":
        expected = revision_payload.get("correct_answer")
        if not expected or not answer_text:
            return _ZERO
        return (
            _ONE
            if _normalize_text(answer_text) == _normalize_text(str(expected))
            else _ZERO
        )
    if qtype == "fill_blank":
        expected = revision_payload.get("correct_answer")
        submitted = _parse_fill_blank_submission(answer_text)
        if not isinstance(expected, list) or len(expected) != len(submitted):
            return _ZERO
        return (
            _ONE
            if all(
                _normalize_text(s) == _normalize_text(str(e))
                for s, e in zip(submitted, expected, strict=True)
            )
            else _ZERO
        )
    snapshot_question = cast(
        QuizQuestion,
        SimpleNamespace(
            numeric_answer=revision_payload.get("numeric_answer"),
            numeric_tolerance=revision_payload.get("numeric_tolerance"),
            match_pairs=revision_payload.get("match_pairs"),
            ordering_sequence=revision_payload.get("ordering_sequence"),
        ),
    )
    if qtype == "numerical":
        return _grade_numerical(snapshot_question, answer_text)
    if qtype == "matching":
        return _grade_matching(snapshot_question, answer_text)
    if qtype == "ordering":
        return _grade_ordering(snapshot_question, answer_text)
    return _ZERO  # code + unknown remain manual-only


_OPEN_RESPONSE_ON_MISS = {"short_answer", "fill_blank"}


def needs_manual_grade(question_type: str, result: GradeResult) -> bool:
    """Return True if this answer must be reviewed by a human (Phase 4).

    Policy:
      - code            -> always (no auto-grader exists; it always scores 0)
      - short_answer    -> only when the auto-grade missed (exact-match failed)
      - fill_blank      -> only when the auto-grade missed
      - multiple_choice / true_false -> never (fully auto-graded)
    """
    if question_type == "code":
        return True
    if question_type in _OPEN_RESPONSE_ON_MISS:
        return not result.is_correct
    return False


__all__ = [
    "GradeResult",
    "grade_answer",
    "grade_answer_against_revision",
    "needs_manual_grade",
]
