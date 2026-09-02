"""Question-bank service (cross-quiz reuse).

Two operations:

* :func:`list_bank_entries` — query published + draft + edited
  questions across the courses an actor can manage, with optional
  filters by module / lesson / type / bloom / difficulty / search.

* :func:`import_questions` — clone a list of source questions into a
  target quiz. Each clone gets a new id, fresh review state, and a
  back-pointer to the source via ``imported_from_question_id``.

The list endpoint scopes by ``course_id`` (the top-level path
parameter) — the router enforces course-update permission before
calling. The service trusts that and never crosses the course
boundary itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.pagination import (
    CursorPage,
    decode_composite_cursor,
    encode_composite_cursor,
)
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.models import Module
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
    QuizQuestionBankItem,
    QuizQuestionBankOption,
    QuizQuestionOption,
    QuizSourceLesson,
)


async def _load_options_for_questions(
    db: AsyncSession, question_ids: list[UUID]
) -> dict[UUID, list[QuizQuestionOption]]:
    """Hydrate the options for every question id in one round-trip.

    ``QuizQuestion`` has no ``options`` relationship in the ORM (kept
    minimal to avoid lazy-load surprises), so the bank service hits
    ``QuizQuestionOption`` directly and groups by ``question_id``.
    """
    if not question_ids:
        return {}
    rows = (
        (
            await db.execute(
                select(QuizQuestionOption)
                .where(QuizQuestionOption.question_id.in_(question_ids))
                .where(QuizQuestionOption.deleted_at.is_(None))
                .order_by(QuizQuestionOption.position)
            )
        )
        .scalars()
        .all()
    )
    grouped: dict[UUID, list[QuizQuestionOption]] = {}
    for option in rows:
        grouped.setdefault(option.question_id, []).append(option)
    return grouped


async def _load_options_for_bank_items(
    db: AsyncSession, item_ids: list[UUID]
) -> dict[UUID, list[QuizQuestionBankOption]]:
    if not item_ids:
        return {}
    rows = list(
        (
            await db.execute(
                select(QuizQuestionBankOption)
                .where(QuizQuestionBankOption.bank_item_id.in_(item_ids))
                .where(QuizQuestionBankOption.deleted_at.is_(None))
                .order_by(QuizQuestionBankOption.position)
            )
        )
        .scalars()
        .all()
    )
    grouped: dict[UUID, list[QuizQuestionBankOption]] = {}
    for option in rows:
        grouped.setdefault(option.bank_item_id, []).append(option)
    return grouped


_QUESTION_CONTENT_FIELDS = (
    "learning_outcome_id",
    "question_type",
    "prompt_text",
    "hint_text",
    "explanation",
    "difficulty",
    "bloom_level",
    "expected_response_time_ms",
    "expected_ef_ceiling",
    "source_refs",
    "original_generated_payload",
    "prompt_format",
    "hint_format",
    "explanation_format",
    "single_answer",
    "answer_numbering",
    "numeric_answer",
    "numeric_tolerance",
    "match_pairs",
    "match_distractors",
    "ordering_sequence",
    "category_id",
)

_OPTION_CONTENT_FIELDS = (
    "option_key",
    "option_text",
    "is_correct",
    "position",
    "option_format",
    "grade_fraction",
    "feedback_text",
    "feedback_format",
)


def _plain_copy(value: Any) -> Any:  # noqa: ANN401 -- arbitrary JSON/Decimal content
    if isinstance(value, dict):
        return {key: _plain_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_copy(item) for item in value]
    return value


def _portable_question_content(source: Any) -> dict[str, Any]:  # noqa: ANN401
    return {field: _plain_copy(getattr(source, field, None)) for field in _QUESTION_CONTENT_FIELDS}


def _portable_option_content(source: Any) -> dict[str, Any]:  # noqa: ANN401
    return {field: _plain_copy(getattr(source, field, None)) for field in _OPTION_CONTENT_FIELDS}


def _content_hash(content: dict[str, Any], options: list[dict[str, Any]]) -> str:
    canonical = {
        **content,
        "prompt_text": str(content.get("prompt_text") or "").strip(),
        "options": sorted(options, key=lambda option: int(option.get("position") or 0)),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def _lock_question_append(db: AsyncSession, quiz_id: UUID) -> None:
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"quiz_question_append:{quiz_id}"},
    )


def _assert_target_editable(quiz: Quiz) -> None:
    if quiz.status == "published":
        raise ConflictError(
            "quiz_published_readonly: a published quiz cannot accept imported questions"
        )


async def _clone_question_into_quiz(
    db: AsyncSession,
    *,
    source: QuizQuestion | QuizQuestionBankItem,
    source_options: list[QuizQuestionOption] | list[QuizQuestionBankOption],
    target_quiz_id: UUID,
    position: int,
    actor: CurrentUser,
    source_question_id: UUID | None = None,
    source_bank_item_id: UUID | None = None,
) -> QuizQuestion:
    """Canonical deep copy used by both legacy and curated-bank imports."""
    content = _portable_question_content(source)
    clone = QuizQuestion(
        quiz_id=target_quiz_id,
        position=position,
        **content,
        review_status="pending",
        imported_from_question_id=source_question_id,
        imported_from_bank_item_id=source_bank_item_id,
        reviewed_by=None,
        reviewed_at=None,
        published_at=None,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(clone)
    await db.flush()
    for option in source_options:
        db.add(
            QuizQuestionOption(
                question_id=clone.id,
                **_portable_option_content(option),
                created_by=actor.user_id,
                updated_by=actor.user_id,
            )
        )
    await db.flush()
    return clone


async def list_bank_entries(  # noqa: C901, PLR0913 -- explicit filter composition
    db: AsyncSession,
    *,
    course_id: UUID,
    module_id: UUID | None = None,
    lesson_id: UUID | None = None,
    question_type: str | None = None,
    bloom_level: str | None = None,
    difficulty: str | None = None,
    review_status: str | None = "approved",
    search: str | None = None,
    exclude_quiz_id: UUID | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> CursorPage[dict[str, Any]]:
    """Cursor-paginated bank rows ordered by ``(updated_at DESC, id DESC)``.

    ``review_status`` defaults to ``approved`` so teachers see only
    vetted questions; pass ``None`` to include every state. Soft-
    deleted rows are filtered automatically by SoftDeleteMixin.

    The result wraps a list of plain dicts shaped to feed
    :class:`QuestionBankEntry` directly. ``cursor`` is opaque and
    round-trips through subsequent calls.
    """
    if limit < 1 or limit > 200:
        raise AppError("limit must be between 1 and 200")

    after_updated_at: datetime | None = None
    after_id: UUID | None = None
    if cursor:
        sort_value, last_id = decode_composite_cursor(cursor)
        if not isinstance(sort_value, datetime):
            raise AppError("Invalid cursor")
        after_updated_at = sort_value
        after_id = last_id

    stmt = (
        select(QuizQuestion, Quiz, Module)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .join(Module, Module.id == Quiz.module_id)
        .where(Quiz.course_id == course_id)
        .where(QuizQuestion.deleted_at.is_(None))
        .where(Quiz.deleted_at.is_(None))
        .order_by(QuizQuestion.updated_at.desc(), QuizQuestion.id.desc())
        .limit(limit)
    )

    if after_updated_at is not None and after_id is not None:
        stmt = stmt.where(
            tuple_(QuizQuestion.updated_at, QuizQuestion.id) < (after_updated_at, after_id)
        )

    if module_id is not None:
        stmt = stmt.where(Quiz.module_id == module_id)
    if lesson_id is not None:
        # quizzes link to lessons via QuizSourceLesson; restrict to
        # quizzes that source the requested lesson.
        stmt = stmt.where(
            QuizQuestion.quiz_id.in_(
                select(QuizSourceLesson.quiz_id).where(QuizSourceLesson.lesson_id == lesson_id)
            )
        )
    if question_type is not None:
        stmt = stmt.where(QuizQuestion.question_type == question_type)
    if bloom_level is not None:
        stmt = stmt.where(QuizQuestion.bloom_level == bloom_level)
    if difficulty is not None:
        stmt = stmt.where(QuizQuestion.difficulty == difficulty)
    if review_status is not None:
        stmt = stmt.where(QuizQuestion.review_status == review_status)
    if exclude_quiz_id is not None:
        stmt = stmt.where(QuizQuestion.quiz_id != exclude_quiz_id)
    if search:
        like = f"%{search.strip()}%"
        stmt = stmt.where(
            or_(
                QuizQuestion.prompt_text.ilike(like),
                Quiz.title.ilike(like),
            )
        )

    rows = (await db.execute(stmt)).all()
    questions = [row[0] for row in rows]
    options_by_question = await _load_options_for_questions(db, [q.id for q in questions])
    for question in questions:
        # Pydantic from_attributes reads the dynamic attribute we set
        # here so the bank entry serialiser sees options without the
        # ORM having a real relationship.
        setattr(question, "options", options_by_question.get(question.id, []))  # noqa: B010 -- dynamic attr
    items = [
        {
            "question": question,
            "quiz_id": quiz.id,
            "quiz_title": quiz.title,
            "module_id": module.id,
            "module_title": module.title,
            "course_id": quiz.course_id,
        }
        for question, quiz, module in rows
    ]
    next_cursor = (
        encode_composite_cursor(questions[-1].updated_at, questions[-1].id)
        if len(questions) == limit
        else None
    )
    return CursorPage(items=items, next_cursor=next_cursor)


async def import_questions(
    db: AsyncSession,
    *,
    target_quiz_id: UUID,
    source_question_ids: list[UUID],
    actor: CurrentUser,
) -> list[QuizQuestion]:
    """Clone the given source questions into ``target_quiz_id``.

    Each clone gets:

    * new ``id`` (autogen via UUIDPrimaryKeyMixin)
    * appended ``position`` (continues the existing series)
    * ``review_status='pending'``, ``reviewed_*`` cleared
    * ``imported_from_question_id`` set to the source row's id
    * cloned options (new ids, same ``option_key`` / text /
      ``is_correct`` / ``position``)

    The router checks course-update permission before delegating; the
    service additionally validates that every source question lives in
    the same course as the target so a teacher can't smuggle a row
    from a course they don't own.
    """
    target_quiz = (
        await db.execute(select(Quiz).where(Quiz.id == target_quiz_id))
    ).scalar_one_or_none()
    if target_quiz is None or target_quiz.deleted_at is not None:
        raise NotFoundError(f"Quiz {target_quiz_id} not found")
    _assert_target_editable(target_quiz)

    if not source_question_ids:
        raise AppError("source_question_ids must not be empty")

    sources = (
        (
            await db.execute(
                select(QuizQuestion)
                .where(QuizQuestion.id.in_(source_question_ids))
                .where(QuizQuestion.deleted_at.is_(None))
            )
        )
        .scalars()
        .all()
    )
    options_by_question = await _load_options_for_questions(db, [q.id for q in sources])

    found_ids = {q.id for q in sources}
    missing = [qid for qid in source_question_ids if qid not in found_ids]
    if missing:
        raise NotFoundError(f"Source question(s) not found: {', '.join(str(m) for m in missing)}")

    unapproved = [question.id for question in sources if question.review_status != "approved"]
    if unapproved:
        raise ConflictError(
            "Only approved source questions can be imported: "
            + ", ".join(str(question_id) for question_id in unapproved)
        )

    # Course-scope guard: every source must belong to the same course
    # as the target. Walk via QuizQuestion → Quiz.course_id.
    source_quiz_ids = {q.quiz_id for q in sources}
    quiz_courses: dict[UUID, UUID] = {
        row.id: row.course_id
        for row in (
            await db.execute(select(Quiz.id, Quiz.course_id).where(Quiz.id.in_(source_quiz_ids)))
        ).all()
    }
    foreign = [qid for qid, course_id in quiz_courses.items() if course_id != target_quiz.course_id]
    if foreign:
        raise AppError(
            "Cannot import questions across course boundaries: "
            f"sources in quizzes {foreign} belong to a different course"
        )

    await _lock_question_append(db, target_quiz_id)
    next_position = await _next_position(db, target_quiz_id)
    cloned: list[QuizQuestion] = []
    sources_by_id = {q.id: q for q in sources}
    for src_id in source_question_ids:  # preserve caller order
        source = sources_by_id[src_id]
        clone = await _clone_question_into_quiz(
            db,
            source=source,
            source_options=options_by_question.get(source.id, []),
            target_quiz_id=target_quiz_id,
            position=next_position,
            actor=actor,
            source_question_id=source.id,
        )
        cloned.append(clone)
        next_position += 1

    await db.flush()
    cloned_options = await _load_options_for_questions(db, [c.id for c in cloned])
    for clone in cloned:
        # ``QuizQuestion`` has no ``options`` ORM relationship; assign as
        # a dynamic attr so Pydantic from_attributes finds it for the
        # router response.
        setattr(clone, "options", cloned_options.get(clone.id, []))  # noqa: B010 -- dynamic, not column
    return cloned


async def _next_position(db: AsyncSession, quiz_id: UUID) -> int:
    """Return the next ``position`` value for a quiz's question list."""
    stmt = select(func.coalesce(func.max(QuizQuestion.position), 0) + 1).where(
        and_(QuizQuestion.quiz_id == quiz_id, QuizQuestion.deleted_at.is_(None))
    )
    return int((await db.execute(stmt)).scalar_one())


async def duplicate_question(
    db: AsyncSession,
    *,
    question_id: UUID,
    actor: CurrentUser,
) -> QuizQuestion:
    """Clone a single question **in place** within its own quiz.

    Unlike :func:`import_questions` (which copies *bank* questions into a
    *different* target quiz), this is the per-question "Duplicate" action in
    the editor: the copy lands at the end of the same quiz's question list.

    The clone gets:

    * a new ``id`` (autogen)
    * ``position`` appended after the current last question
    * ``review_status='pending'`` with ``reviewed_by`` / ``reviewed_at`` /
      ``published_at`` cleared — a duplicate is unvetted content and must
      re-enter the review queue, never inherit the source's approval.
    * ``imported_from_question_id`` pointing back at the source
    * every option cloned (new ids, same key/text/is_correct/position and the
      Phase-3/7/8 extras: formats, grade_fraction, feedback).

    Runtime rows (attempts, answers, revisions) are intentionally NOT copied.
    """
    source = (
        await db.execute(
            select(QuizQuestion)
            .where(QuizQuestion.id == question_id)
            .where(QuizQuestion.deleted_at.is_(None))
        )
    ).scalar_one_or_none()
    if source is None:
        raise NotFoundError(f"Question {question_id} not found")

    options = (await _load_options_for_questions(db, [source.id])).get(source.id, [])

    clone = QuizQuestion(
        quiz_id=source.quiz_id,
        learning_outcome_id=source.learning_outcome_id,
        position=await _next_position(db, source.quiz_id),
        question_type=source.question_type,
        prompt_text=source.prompt_text,
        hint_text=source.hint_text,
        explanation=source.explanation,
        difficulty=source.difficulty,
        bloom_level=source.bloom_level,
        review_status="pending",
        expected_response_time_ms=source.expected_response_time_ms,
        expected_ef_ceiling=source.expected_ef_ceiling,
        source_refs=list(source.source_refs or []),
        original_generated_payload=(
            dict(source.original_generated_payload) if source.original_generated_payload else None
        ),
        imported_from_question_id=source.id,
        prompt_format=source.prompt_format,
        hint_format=source.hint_format,
        explanation_format=source.explanation_format,
        single_answer=source.single_answer,
        answer_numbering=source.answer_numbering,
        numeric_answer=source.numeric_answer,
        numeric_tolerance=source.numeric_tolerance,
        match_pairs=(list(source.match_pairs) if source.match_pairs is not None else None),
        ordering_sequence=(
            list(source.ordering_sequence) if source.ordering_sequence is not None else None
        ),
        category_id=source.category_id,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(clone)
    await db.flush()  # populate clone.id for option FK

    for option in options:
        db.add(
            QuizQuestionOption(
                question_id=clone.id,
                option_key=option.option_key,
                option_text=option.option_text,
                is_correct=option.is_correct,
                position=option.position,
                option_format=option.option_format,
                grade_fraction=option.grade_fraction,
                feedback_text=option.feedback_text,
                feedback_format=option.feedback_format,
                created_by=actor.user_id,
                updated_by=actor.user_id,
            )
        )
    await db.flush()

    cloned_options = (await _load_options_for_questions(db, [clone.id])).get(clone.id, [])
    setattr(clone, "options", cloned_options)  # noqa: B010 -- dynamic, not a column
    return clone


__all__ = [
    "duplicate_question",
    "import_questions",
    "list_bank_entries",
]
