"""Quiz question import/export service (Phase 11).

Bridges the pure format parsers/serializers (``services/formats/``) to the DB:

* :func:`import_questions_from_file` — parse a GIFT / Moodle-XML file, then create
  each supported question via ``authoring.create_question`` (so revisions/audit
  fire consistently). Unsupported types are collected as warnings, never fatal.
  A structurally broken file aborts with no writes (the parser raises).
* :func:`export_quiz_questions` — serialize a quiz's questions to GIFT or XML.

Type mapping and the additive/review-gated import policy follow the Phase 11
decision record. Teacher-only; export never leaves a learner route.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.exceptions import AppError
from abridgeai.features.quizzes.models import QuizQuestion, QuizQuestionOption
from abridgeai.features.quizzes.services import authoring as _authoring
from abridgeai.features.quizzes.services.formats._types import (
    ParsedOption,
    ParsedQuestion,
)
from abridgeai.features.quizzes.services.formats.gift import parse_gift, serialize_gift
from abridgeai.features.quizzes.services.formats.moodle_xml import (
    parse_moodle_xml,
    serialize_moodle_xml,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.core.security import CurrentUser

_OPTION_KEYS = "ABCDEFGHIJ"


def _parse(content: str, fmt: str):
    if fmt == "gift":
        return parse_gift(content)
    if fmt == "xml":
        return parse_moodle_xml(content)
    raise AppError(f"Unsupported import format: {fmt}")


def _to_create_payload(q: ParsedQuestion) -> SimpleNamespace:
    """Build a create_question-compatible payload from a parsed question."""
    options = []
    for idx, opt in enumerate(q.options):
        options.append(
            SimpleNamespace(
                option_key=_OPTION_KEYS[idx] if idx < len(_OPTION_KEYS) else str(idx + 1),
                option_text=opt.text,
                is_correct=opt.is_correct,
            )
        )
    return SimpleNamespace(
        question_type=q.question_type,
        prompt_text=q.prompt_text,
        hint_text=None,
        explanation=q.explanation,
        options=options,
        review_status="pending",
        # short_answer/fill_blank carry the expected answer on the payload the
        # grader reads via original_generated_payload; create_question copies it.
        correct_answer=q.correct_answer,
    )


async def import_questions_from_file(
    db: AsyncSession,
    *,
    quiz_id: UUID,
    content: str,
    fmt: str,
    actor: CurrentUser,
) -> dict[str, Any]:
    """Parse ``content`` and create supported questions on ``quiz_id``.

    Returns ``{"imported": int, "skipped": int, "warnings": [...]}``. A malformed
    file raises (parser error → 422 upstream); per-question issues are warnings.
    """
    result = _parse(content, fmt)
    warnings = list(result.warnings)
    imported = 0
    for i, q in enumerate(result.questions, start=1):
        # true_false options need T/F keys; create_question validates them.
        payload = _to_create_payload(q)
        if q.question_type == "true_false":
            for opt, key in zip(payload.options, ["T", "F"], strict=False):
                opt.option_key = key
        try:
            created = await _authoring.create_question(db, quiz_id, payload, actor)
            # Persist the expected answer for open-response types so the grader
            # can score them (create_question sets original_generated_payload=None).
            if q.correct_answer is not None:
                created.original_generated_payload = {"correct_answer": q.correct_answer}
            imported += 1
        except AppError as exc:
            warnings.append(f"Q{i}: {exc}")
    return {
        "imported": imported,
        "skipped": len(result.questions) - imported,
        "warnings": warnings,
    }


async def export_quiz_questions(
    db: AsyncSession, *, quiz_id: UUID, fmt: str
) -> str:
    """Serialize a quiz's questions to GIFT or Moodle XML (teacher-only)."""
    from sqlalchemy import select  # noqa: PLC0415

    questions = (
        await db.execute(
            select(QuizQuestion)
            .where(QuizQuestion.quiz_id == quiz_id, QuizQuestion.deleted_at.is_(None))
            .order_by(QuizQuestion.position)
        )
    ).scalars().all()
    options_by_q: dict[UUID, list[QuizQuestionOption]] = {}
    if questions:
        opt_rows = (
            await db.execute(
                select(QuizQuestionOption)
                .where(QuizQuestionOption.question_id.in_([q.id for q in questions]))
                .order_by(QuizQuestionOption.position)
            )
        ).scalars().all()
        for o in opt_rows:
            options_by_q.setdefault(o.question_id, []).append(o)

    parsed: list[ParsedQuestion] = []
    for q in questions:
        payload = q.original_generated_payload or {}
        parsed.append(
            ParsedQuestion(
                question_type=q.question_type,
                prompt_text=q.prompt_text,
                options=[
                    ParsedOption(text=o.option_text, is_correct=o.is_correct)
                    for o in options_by_q.get(q.id, [])
                ],
                correct_answer=(
                    payload.get("correct_answer")
                    if isinstance(payload, dict)
                    else None
                ),
                explanation=q.explanation,
            )
        )
    if fmt == "gift":
        return serialize_gift(parsed)
    if fmt == "xml":
        return serialize_moodle_xml(parsed)
    raise AppError(f"Unsupported export format: {fmt}")


__all__ = ["export_quiz_questions", "import_questions_from_file"]
