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

from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.models import Module
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
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
        await db.execute(
            select(QuizQuestionOption)
            .where(QuizQuestionOption.question_id.in_(question_ids))
            .where(QuizQuestionOption.deleted_at.is_(None))
            .order_by(QuizQuestionOption.position)
        )
    ).scalars().all()
    grouped: dict[UUID, list[QuizQuestionOption]] = {}
    for option in rows:
        grouped.setdefault(option.question_id, []).append(option)
    return grouped


async def list_bank_entries(  # noqa: PLR0913 -- filter knobs are intentionally explicit
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
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return scoped bank rows ordered by recency.

    ``review_status`` defaults to ``approved`` so teachers see only
    vetted questions; pass ``None`` to include every state. Soft-
    deleted rows are filtered automatically by SoftDeleteMixin.

    The result is a list of plain dicts shaped to feed
    :class:`QuestionBankEntry` directly.
    """
    if limit < 1 or limit > 200:
        raise AppError("limit must be between 1 and 200")
    if offset < 0:
        raise AppError("offset must be non-negative")

    stmt = (
        select(QuizQuestion, Quiz, Module)
        .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
        .join(Module, Module.id == Quiz.module_id)
        .where(Quiz.course_id == course_id)
        .where(QuizQuestion.deleted_at.is_(None))
        .where(Quiz.deleted_at.is_(None))
        .order_by(QuizQuestion.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )

    if module_id is not None:
        stmt = stmt.where(Quiz.module_id == module_id)
    if lesson_id is not None:
        # quizzes link to lessons via QuizSourceLesson; restrict to
        # quizzes that source the requested lesson.
        stmt = stmt.where(
            QuizQuestion.quiz_id.in_(
                select(QuizSourceLesson.quiz_id).where(
                    QuizSourceLesson.lesson_id == lesson_id
                )
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
    options_by_question = await _load_options_for_questions(
        db, [q.id for q in questions]
    )
    for question in questions:
        # Pydantic from_attributes reads the dynamic attribute we set
        # here so the bank entry serialiser sees options without the
        # ORM having a real relationship.
        setattr(question, "options", options_by_question.get(question.id, []))  # noqa: B010 -- dynamic attr
    return [
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

    if not source_question_ids:
        raise AppError("source_question_ids must not be empty")

    sources = (
        await db.execute(
            select(QuizQuestion)
            .where(QuizQuestion.id.in_(source_question_ids))
            .where(QuizQuestion.deleted_at.is_(None))
        )
    ).scalars().all()
    options_by_question = await _load_options_for_questions(
        db, [q.id for q in sources]
    )

    found_ids = {q.id for q in sources}
    missing = [qid for qid in source_question_ids if qid not in found_ids]
    if missing:
        raise NotFoundError(
            f"Source question(s) not found: {', '.join(str(m) for m in missing)}"
        )

    # Course-scope guard: every source must belong to the same course
    # as the target. Walk via QuizQuestion → Quiz.course_id.
    source_quiz_ids = {q.quiz_id for q in sources}
    quiz_courses: dict[UUID, UUID] = {
        row.id: row.course_id
        for row in (
            await db.execute(
                select(Quiz.id, Quiz.course_id).where(Quiz.id.in_(source_quiz_ids))
            )
        ).all()
    }
    foreign = [
        qid
        for qid, course_id in quiz_courses.items()
        if course_id != target_quiz.course_id
    ]
    if foreign:
        raise AppError(
            "Cannot import questions across course boundaries: "
            f"sources in quizzes {foreign} belong to a different course"
        )

    next_position = await _next_position(db, target_quiz_id)
    cloned: list[QuizQuestion] = []
    sources_by_id = {q.id: q for q in sources}
    for src_id in source_question_ids:  # preserve caller order
        source = sources_by_id[src_id]
        clone = QuizQuestion(
            quiz_id=target_quiz_id,
            position=next_position,
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
                dict(source.original_generated_payload)
                if source.original_generated_payload
                else None
            ),
            imported_from_question_id=source.id,
            created_by=actor.user_id,
            updated_by=actor.user_id,
        )
        db.add(clone)
        await db.flush()  # populate clone.id for option FK
        for option in options_by_question.get(source.id, []):
            db.add(
                QuizQuestionOption(
                    question_id=clone.id,
                    option_key=option.option_key,
                    option_text=option.option_text,
                    is_correct=option.is_correct,
                    position=option.position,
                    created_by=actor.user_id,
                    updated_by=actor.user_id,
                )
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


__all__ = ["import_questions", "list_bank_entries"]
