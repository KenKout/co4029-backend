"""Lesson unlock gate per thesis §5.x + §3.2 — combined EF + interview + prereqs.

Cross-feature contract: the gate reads from ``lessons``,
``lesson_prerequisites``, ``module_items``, ``modules``, ``quizzes``,
``quiz_questions``, ``student_card_state`` and ``interview_sessions``
through raw ``sqlalchemy.text(...)``. We do NOT import models from
``features.courses`` / ``features.quizzes`` / ``features.interviews``
to keep the Features-independent import-linter contract intact; the
module is whitelisted in ``[tool.importlinter]`` ``ignore_imports`` for
the "services do not touch SQLAlchemy" contract on the same grounds as
``services/review.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.cache.decorators import cached
from abridgeai.core.cache.keys import LESSON_UNLOCK
from abridgeai.features.spaced_repetition.queries import (
    DEFAULT_BLOCKING_LIMIT,
    aggregate_lesson_card_ef,
    fetch_lesson_module_id,
    fetch_lesson_unlock_config,
    fetch_prerequisite_lesson_ids,
    has_passing_interview_for_module,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockingCardInfo:
    """A card whose stored EF is below the lesson threshold."""

    question_id: UUID
    current_ef: float
    quiz_id: UUID
    source_chunk_ids: list[UUID] = field(default_factory=list)


@dataclass(frozen=True)
class LessonUnlockStatus:
    """Combined unlock-gate verdict for a (student, lesson) pair."""

    eligible: bool
    current_ratio: float
    required_ratio: float
    ef_min: float
    total_cards: int
    passing_cards: int
    blocking_cards: list[BlockingCardInfo]
    prereq_lesson_ids_unlocked: bool
    interview_pass_required: bool
    interview_passed: bool
    next_unlock_estimate: str | None = None


def _coerce_uuid_list(raw: object) -> list[UUID]:
    if not isinstance(raw, list):
        return []
    out: list[UUID] = []
    for item in raw:
        if isinstance(item, UUID):
            out.append(item)
            continue
        try:
            out.append(UUID(str(item)))
        except (ValueError, TypeError):
            continue
    return out


def _coerce_uuid(raw: object) -> UUID | None:
    if isinstance(raw, UUID):
        return raw
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return None


def _format_estimate(
    *,
    eligible: bool,
    prereqs_ok: bool,
    interview_required: bool,
    interview_passed: bool,
    passing: int,
    total: int,
    required_ratio: float,
) -> str | None:
    if eligible:
        return None
    if not prereqs_ok:
        return "Complete the prerequisite lessons first."
    if interview_required and not interview_passed:
        return "Pass the module interview to unlock this lesson."
    if total == 0:
        return None
    needed = max(int((required_ratio * total) + 0.999999) - passing, 0)
    if needed <= 0:
        return None
    cards_word = "card" if needed == 1 else "cards"
    return f"Review {needed} more {cards_word} to reach the unlock threshold."


def _status_to_dict(status: LessonUnlockStatus) -> dict[str, Any]:
    return {
        "eligible": status.eligible,
        "current_ratio": status.current_ratio,
        "required_ratio": status.required_ratio,
        "ef_min": status.ef_min,
        "total_cards": status.total_cards,
        "passing_cards": status.passing_cards,
        "blocking_cards": [
            {
                "question_id": str(b.question_id),
                "current_ef": b.current_ef,
                "quiz_id": str(b.quiz_id),
                "source_chunk_ids": [str(s) for s in b.source_chunk_ids],
            }
            for b in status.blocking_cards
        ],
        "prereq_lesson_ids_unlocked": status.prereq_lesson_ids_unlocked,
        "interview_pass_required": status.interview_pass_required,
        "interview_passed": status.interview_passed,
        "next_unlock_estimate": status.next_unlock_estimate,
    }


def _status_from_dict(payload: dict[str, Any]) -> LessonUnlockStatus:
    blocking_raw = payload.get("blocking_cards") or []
    blocking_cards: list[BlockingCardInfo] = []
    for entry in blocking_raw:
        if not isinstance(entry, dict):
            continue
        qid = _coerce_uuid(entry.get("question_id"))
        quiz_id = _coerce_uuid(entry.get("quiz_id"))
        if qid is None or quiz_id is None:
            continue
        try:
            current_ef = float(entry.get("current_ef") or 0.0)
        except (TypeError, ValueError):
            current_ef = 0.0
        blocking_cards.append(
            BlockingCardInfo(
                question_id=qid,
                current_ef=current_ef,
                quiz_id=quiz_id,
                source_chunk_ids=_coerce_uuid_list(entry.get("source_chunk_ids")),
            )
        )
    return LessonUnlockStatus(
        eligible=bool(payload.get("eligible", False)),
        current_ratio=float(payload.get("current_ratio", 0.0)),
        required_ratio=float(payload.get("required_ratio", 0.0)),
        ef_min=float(payload.get("ef_min", 0.0)),
        total_cards=int(payload.get("total_cards", 0)),
        passing_cards=int(payload.get("passing_cards", 0)),
        blocking_cards=blocking_cards,
        prereq_lesson_ids_unlocked=bool(payload.get("prereq_lesson_ids_unlocked", False)),
        interview_pass_required=bool(payload.get("interview_pass_required", False)),
        interview_passed=bool(payload.get("interview_passed", False)),
        next_unlock_estimate=payload.get("next_unlock_estimate"),
    )


async def _prereqs_unlocked(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
    visited: set[UUID],
) -> bool:
    if lesson_id in visited:
        logger.warning(
            "lesson_unlock.cycle_detected",
            extra={
                "event": "lesson_unlock_cycle_detected",
                "lesson_id": str(lesson_id),
                "student_id": str(student_id),
                "visited": sorted(str(v) for v in visited),
            },
        )
        return True
    visited.add(lesson_id)

    prereq_ids = await fetch_prerequisite_lesson_ids(db, lesson_id=lesson_id)
    for prereq_id in prereq_ids:
        status = await _check_unlock_recursive(
            db,
            student_id=student_id,
            lesson_id=prereq_id,
            visited=visited,
        )
        if not status.eligible:
            return False
    return True


async def _check_unlock_recursive(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
    visited: set[UUID],
) -> LessonUnlockStatus:
    config = await fetch_lesson_unlock_config(db, lesson_id=lesson_id)
    if config is None:
        return LessonUnlockStatus(
            eligible=False,
            current_ratio=0.0,
            required_ratio=0.0,
            ef_min=0.0,
            total_cards=0,
            passing_cards=0,
            blocking_cards=[],
            prereq_lesson_ids_unlocked=False,
            interview_pass_required=False,
            interview_passed=False,
            next_unlock_estimate="Lesson not found.",
        )
    ef_min, tau_unlock, requires_interview_pass = config

    prereqs_ok = await _prereqs_unlocked(
        db, student_id=student_id, lesson_id=lesson_id, visited=visited
    )

    passing, total, blocking_payload = await aggregate_lesson_card_ef(
        db,
        student_id=student_id,
        lesson_id=lesson_id,
        ef_min=ef_min,
        blocking_limit=DEFAULT_BLOCKING_LIMIT,
    )
    current_ratio = (passing / total) if total > 0 else 0.0

    interview_passed = False
    if requires_interview_pass:
        module_id = await fetch_lesson_module_id(db, lesson_id=lesson_id)
        if module_id is not None:
            interview_passed = await has_passing_interview_for_module(
                db, student_id=student_id, module_id=module_id
            )

    ef_gate_ok = total == 0 or current_ratio >= tau_unlock
    interview_gate_ok = (not requires_interview_pass) or interview_passed
    eligible = prereqs_ok and ef_gate_ok and interview_gate_ok

    blocking_cards: list[BlockingCardInfo] = []
    for entry in blocking_payload:
        if not isinstance(entry, dict):
            continue
        qid = _coerce_uuid(entry.get("question_id"))
        quiz_id = _coerce_uuid(entry.get("quiz_id"))
        if qid is None or quiz_id is None:
            continue
        try:
            current_ef = float(entry.get("current_ef") or 0.0)
        except (TypeError, ValueError):
            current_ef = 0.0
        blocking_cards.append(
            BlockingCardInfo(
                question_id=qid,
                current_ef=current_ef,
                quiz_id=quiz_id,
                source_chunk_ids=_coerce_uuid_list(entry.get("source_chunk_ids")),
            )
        )

    estimate = _format_estimate(
        eligible=eligible,
        prereqs_ok=prereqs_ok,
        interview_required=requires_interview_pass,
        interview_passed=interview_passed,
        passing=passing,
        total=total,
        required_ratio=tau_unlock,
    )

    return LessonUnlockStatus(
        eligible=eligible,
        current_ratio=current_ratio,
        required_ratio=tau_unlock,
        ef_min=ef_min,
        total_cards=total,
        passing_cards=passing,
        blocking_cards=blocking_cards,
        prereq_lesson_ids_unlocked=prereqs_ok,
        interview_pass_required=requires_interview_pass,
        interview_passed=interview_passed,
        next_unlock_estimate=estimate,
    )


@cached(LESSON_UNLOCK)
async def _check_lesson_unlock_cached(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
) -> dict[str, Any]:
    status = await _check_unlock_recursive(
        db, student_id=student_id, lesson_id=lesson_id, visited=set()
    )
    return _status_to_dict(status)


async def check_lesson_unlock(
    db: AsyncSession,
    *,
    student_id: UUID,
    lesson_id: UUID,
) -> LessonUnlockStatus:
    """Compute the lesson unlock status for a learner.

    Combined gate per thesis §5.x + §3.2:

    * **Prerequisite gate** — every lesson in ``lesson_prerequisites``
      must itself be eligible (recursively, cycle-safe).
    * **EF gate** — ``passing_cards / total_cards >= tau_unlock`` where
      a card is "passing" iff its stored EF is at least
      ``lesson.ef_min_unlock``. Empty lessons (0 cards) bypass the EF
      gate.
    * **Interview gate** — when the lesson sets
      ``requires_interview_pass``, the student must have a completed
      ``interview_sessions`` row with ``pass_verdict = TRUE`` against
      the interview config attached to the lesson's module.

    Cycle-safety: if A→B→A appears in the prereq graph the recursion
    short-circuits with a WARN log and treats the cycle as eligible so
    the student is never bricked out.

    Cached via T0.29 :func:`@cached` decorator with the
    :data:`LESSON_UNLOCK` key (TTL 60 s; auto-invalidated by the
    ``after_flush`` listener once T7.5.13 wires SR write rules). The
    cache stores the dict round-trip form; this wrapper reconstructs
    the dataclass on every call.
    """
    payload = await _check_lesson_unlock_cached(db, student_id=student_id, lesson_id=lesson_id)
    return _status_from_dict(payload)


__all__ = [
    "BlockingCardInfo",
    "LessonUnlockStatus",
    "check_lesson_unlock",
]
