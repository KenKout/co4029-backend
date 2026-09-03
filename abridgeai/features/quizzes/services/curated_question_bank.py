from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.pagination import CursorPage, decode_composite_cursor, encode_composite_cursor
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
    QuizQuestionBankItem,
    QuizQuestionBankOption,
)
from abridgeai.features.quizzes.services.question_bank import (
    _QUESTION_CONTENT_FIELDS,
    _assert_target_editable,
    _clone_question_into_quiz,
    _content_hash,
    _load_options_for_bank_items,
    _load_options_for_questions,
    _lock_question_append,
    _next_position,
    _plain_copy,
    _portable_option_content,
    _portable_question_content,
)

"""Course-scoped curated Quiz Question Bank service."""


@dataclass
class CuratedBankCopyResult:
    """Outcome of a copy-into-bank batch: what got created, what was skipped.

    ``skipped_question_ids`` are the SOURCE Quiz question ids whose content
    already has a live (non-archived) bank copy in the same course. They
    are not copied again; the router surfaces them so the caller can tell
    the teacher exactly which questions already existed.
    """

    created: list[QuizQuestionBankItem]
    skipped_question_ids: list[UUID]


async def _require_course(db: AsyncSession, course_id: UUID) -> None:
    from abridgeai.features.courses.api import public as courses_public  # noqa: PLC0415

    if await courses_public.get_course_by_id(db, course_id) is None:
        raise NotFoundError(f"Course {course_id} not found")


async def _validate_outcome_scope(
    db: AsyncSession, *, course_id: UUID, learning_outcome_id: UUID | None
) -> None:
    if learning_outcome_id is None:
        return
    outcome_course = (
        await db.execute(
            text(
                "SELECT course_id FROM course_learning_outcomes "
                "WHERE id = :outcome_id AND deleted_at IS NULL"
            ),
            {"outcome_id": learning_outcome_id},
        )
    ).scalar_one_or_none()
    if outcome_course != course_id:
        raise AppError("learning_outcome_id does not belong to course")


def _payload_content(payload: Any) -> tuple[dict[str, Any], list[Any]]:  # noqa: ANN401
    data = payload.model_dump(exclude_unset=True)
    options = list(data.pop("options", []) or [])
    data.pop("status", None)
    content = {field: _plain_copy(data.get(field)) for field in _QUESTION_CONTENT_FIELDS}
    content["prompt_text"] = str(content.get("prompt_text") or "").strip()
    content["source_refs"] = content.get("source_refs") or []
    for field, default in (
        ("prompt_format", "plain"),
        ("hint_format", "plain"),
        ("explanation_format", "plain"),
        ("answer_numbering", "abc"),
    ):
        content[field] = content.get(field) or default
    if content.get("single_answer") is None:
        content["single_answer"] = True
    return content, options


def _option_dict(option: Any, *, default_position: int) -> dict[str, Any]:  # noqa: ANN401
    data = option.model_dump() if hasattr(option, "model_dump") else dict(option)
    return {
        "option_key": str(data.get("option_key") or "").strip().upper(),
        "option_text": str(data.get("option_text") or "").strip(),
        "is_correct": bool(data.get("is_correct", False)),
        "position": int(data.get("position") or default_position),
        "option_format": data.get("option_format") or "plain",
        "grade_fraction": data.get("grade_fraction"),
        "feedback_text": data.get("feedback_text"),
        "feedback_format": data.get("feedback_format"),
    }


def _validate_bank_content(content: dict[str, Any], options: list[Any]) -> list[dict[str, Any]]:
    if not content.get("prompt_text"):
        raise AppError("Question text is required")
    normalized = [
        _option_dict(option, default_position=index) for index, option in enumerate(options, 1)
    ]
    if len({option["position"] for option in normalized}) != len(normalized):
        raise AppError("Question option positions must be unique")
    if len({option["option_key"] for option in normalized}) != len(normalized):
        raise AppError("Question option keys must be unique")

    # Reuse the exact type-aware validation used by normal Quiz authoring.
    from abridgeai.features.quizzes.services.authoring import (  # noqa: PLC0415
        _validate_question_options,
    )

    option_shims = [type("Option", (), option)() for option in normalized]
    _validate_question_options(
        str(content.get("question_type")),
        option_shims,
        single_answer=bool(content.get("single_answer", True)),
    )
    return normalized


def _sanitize_bank_content(
    content: dict[str, Any], options: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from abridgeai.core.sanitize import (  # noqa: PLC0415
        sanitize_rich_content,
    )

    sanitized = dict(content)
    for text_field, format_field in (
        ("prompt_text", "prompt_format"),
        ("hint_text", "hint_format"),
        ("explanation", "explanation_format"),
    ):
        sanitized[text_field] = sanitize_rich_content(
            sanitized.get(text_field), fmt=sanitized.get(format_field)
        )
    sanitized_options: list[dict[str, Any]] = []
    for option in options:
        clean_option = dict(option)
        clean_option["option_text"] = sanitize_rich_content(
            clean_option.get("option_text"), fmt=clean_option.get("option_format")
        )
        clean_option["feedback_text"] = sanitize_rich_content(
            clean_option.get("feedback_text"), fmt=clean_option.get("feedback_format")
        )
        sanitized_options.append(clean_option)
    return sanitized, sanitized_options


async def _ensure_unique_bank_content(
    db: AsyncSession,
    *,
    course_id: UUID,
    content_hash: str,
    exclude_item_id: UUID | None = None,
) -> None:
    stmt = select(QuizQuestionBankItem.id).where(
        QuizQuestionBankItem.course_id == course_id,
        QuizQuestionBankItem.content_hash == content_hash,
        QuizQuestionBankItem.deleted_at.is_(None),
        QuizQuestionBankItem.status != "archived",
    )
    if exclude_item_id is not None:
        stmt = stmt.where(QuizQuestionBankItem.id != exclude_item_id)
    if (await db.execute(stmt.limit(1))).scalar_one_or_none() is not None:
        raise ConflictError("quiz_question_bank_duplicate_content")


async def _persist_bank_item(
    db: AsyncSession,
    *,
    course_id: UUID,
    content: dict[str, Any],
    options: list[dict[str, Any]],
    actor: CurrentUser,
    status: str,
    source_question_id: UUID | None = None,
) -> QuizQuestionBankItem:
    await _validate_outcome_scope(
        db, course_id=course_id, learning_outcome_id=content.get("learning_outcome_id")
    )
    digest = _content_hash(content, options)
    await _ensure_unique_bank_content(db, course_id=course_id, content_hash=digest)
    item = QuizQuestionBankItem(
        course_id=course_id,
        source_question_id=source_question_id,
        status=status,
        content_hash=digest,
        **content,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(item)
    await db.flush()
    for option in options:
        db.add(
            QuizQuestionBankOption(
                bank_item_id=item.id,
                **option,
                created_by=actor.user_id,
                updated_by=actor.user_id,
            )
        )
    await flush_or_conflict(db)
    await db.refresh(item)
    item.options = (await _load_options_for_bank_items(db, [item.id])).get(item.id, [])
    return item


async def create_curated_bank_item(
    db: AsyncSession,
    *,
    course_id: UUID,
    payload: Any,  # noqa: ANN401 -- API schema
    actor: CurrentUser,
) -> QuizQuestionBankItem:
    await _require_course(db, course_id)
    if payload.status != "draft":
        raise AppError("Manually created bank questions must start as draft")
    content, raw_options = _payload_content(payload)
    options = _validate_bank_content(content, raw_options)
    content, options = _sanitize_bank_content(content, options)
    return await _persist_bank_item(
        db,
        course_id=course_id,
        content=content,
        options=options,
        actor=actor,
        status="draft",
    )


async def copy_questions_to_curated_bank(
    db: AsyncSession,
    *,
    course_id: UUID,
    question_ids: list[UUID],
    actor: CurrentUser,
) -> CuratedBankCopyResult:
    await _require_course(db, course_id)
    if len(question_ids) != len(set(question_ids)):
        raise AppError("question_ids contains duplicates")
    questions = list(
        (
            await db.execute(
                select(QuizQuestion)
                .join(Quiz, Quiz.id == QuizQuestion.quiz_id)
                .where(
                    QuizQuestion.id.in_(question_ids),
                    QuizQuestion.deleted_at.is_(None),
                    Quiz.deleted_at.is_(None),
                    Quiz.course_id == course_id,
                )
            )
        )
        .scalars()
        .all()
    )
    if len(questions) != len(question_ids):
        raise NotFoundError("One or more Quiz questions were not found in course")
    by_id = {question.id: question for question in questions}
    options_by_question = await _load_options_for_questions(db, question_ids)

    # Snapshot every requested question up front so duplicates can be named
    # in the response. The per-item guard inside ``_persist_bank_item`` stays:
    # it races against a concurrent request landing the same content between
    # this scan and the insert.
    portables: list[tuple[UUID, dict[str, Any], list[dict[str, Any]], str]] = []
    for question_id in question_ids:
        source = by_id[question_id]
        content = _portable_question_content(source)
        option_content = [
            _portable_option_content(option)
            for option in options_by_question.get(question_id, [])
        ]
        _validate_bank_content(content, option_content)
        digest = _content_hash(content, option_content)
        portables.append((question_id, content, option_content, digest))

    existing_hashes = set(
        (
            await db.execute(
                select(QuizQuestionBankItem.content_hash).where(
                    QuizQuestionBankItem.course_id == course_id,
                    QuizQuestionBankItem.content_hash.in_(
                        [digest for _, _, _, digest in portables]
                    ),
                    QuizQuestionBankItem.deleted_at.is_(None),
                    QuizQuestionBankItem.status != "archived",
                )
            )
        )
        .scalars()
        .all()
    )
    skipped_ids = {
        question_id
        for question_id, _, _, digest in portables
        if digest in existing_hashes
    }

    created: list[QuizQuestionBankItem] = []
    for question_id, content, option_content, _ in portables:
        if question_id in skipped_ids:
            continue
        source = by_id[question_id]
        created.append(
            await _persist_bank_item(
                db,
                course_id=course_id,
                content=content,
                options=option_content,
                actor=actor,
                status="approved" if source.review_status == "approved" else "draft",
                source_question_id=source.id,
            )
        )
    return CuratedBankCopyResult(created=created, skipped_question_ids=list(skipped_ids))


async def list_curated_bank_items(  # noqa: PLR0913 -- explicit API filters
    db: AsyncSession,
    *,
    course_id: UUID,
    status: str | None = None,
    question_type: str | None = None,
    bloom_level: str | None = None,
    difficulty: str | None = None,
    search: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> CursorPage[QuizQuestionBankItem]:
    await _require_course(db, course_id)
    if limit < 1 or limit > 200:
        raise AppError("limit must be between 1 and 200")
    stmt = (
        select(QuizQuestionBankItem)
        .where(
            QuizQuestionBankItem.course_id == course_id,
            QuizQuestionBankItem.deleted_at.is_(None),
        )
        .order_by(QuizQuestionBankItem.updated_at.desc(), QuizQuestionBankItem.id.desc())
        .limit(limit)
    )
    if cursor:
        after_updated_at, after_id = decode_composite_cursor(cursor)
        if not isinstance(after_updated_at, datetime):
            raise AppError("Invalid cursor")
        stmt = stmt.where(
            tuple_(QuizQuestionBankItem.updated_at, QuizQuestionBankItem.id)
            < (after_updated_at, after_id)
        )
    if status:
        stmt = stmt.where(QuizQuestionBankItem.status == status)
    if question_type:
        stmt = stmt.where(QuizQuestionBankItem.question_type == question_type)
    if bloom_level:
        stmt = stmt.where(QuizQuestionBankItem.bloom_level == bloom_level)
    if difficulty:
        stmt = stmt.where(QuizQuestionBankItem.difficulty == difficulty)
    if search:
        stmt = stmt.where(QuizQuestionBankItem.prompt_text.ilike(f"%{search.strip()}%"))
    items = list((await db.execute(stmt)).scalars().all())
    options_by_item = await _load_options_for_bank_items(db, [item.id for item in items])
    for item in items:
        item.options = options_by_item.get(item.id, [])
    next_cursor = (
        encode_composite_cursor(items[-1].updated_at, items[-1].id) if len(items) == limit else None
    )
    return CursorPage(items=items, next_cursor=next_cursor)


async def update_curated_bank_item(
    db: AsyncSession,
    *,
    course_id: UUID,
    item_id: UUID,
    payload: Any,  # noqa: ANN401 -- API schema
    actor: CurrentUser,
) -> QuizQuestionBankItem:
    item = (
        await db.execute(
            select(QuizQuestionBankItem)
            .where(
                QuizQuestionBankItem.id == item_id,
                QuizQuestionBankItem.course_id == course_id,
                QuizQuestionBankItem.deleted_at.is_(None),
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Quiz question bank item {item_id} not found")
    existing_options = (await _load_options_for_bank_items(db, [item.id])).get(item.id, [])
    data = payload.model_dump(exclude_unset=True)
    raw_options = data.pop("options", None)
    content = _portable_question_content(item)
    for field, value in data.items():
        if field in _QUESTION_CONTENT_FIELDS:
            content[field] = _plain_copy(value)
    content["prompt_text"] = str(content.get("prompt_text") or "").strip()
    options = (
        _validate_bank_content(content, raw_options)
        if raw_options is not None
        else [_portable_option_content(option) for option in existing_options]
    )
    if raw_options is None:
        _validate_bank_content(content, options)
    content, options = _sanitize_bank_content(content, options)
    await _validate_outcome_scope(
        db, course_id=course_id, learning_outcome_id=content.get("learning_outcome_id")
    )
    digest = _content_hash(content, options)
    await _ensure_unique_bank_content(
        db, course_id=course_id, content_hash=digest, exclude_item_id=item.id
    )
    content_changed = digest != item.content_hash
    for field, value in content.items():
        setattr(item, field, value)
    item.content_hash = digest
    item.updated_by = actor.user_id
    if content_changed and item.status == "approved":
        item.status = "draft"
    if raw_options is not None:
        await db.execute(
            delete(QuizQuestionBankOption).where(QuizQuestionBankOption.bank_item_id == item.id)
        )
        for option in options:
            db.add(
                QuizQuestionBankOption(
                    bank_item_id=item.id,
                    **option,
                    created_by=actor.user_id,
                    updated_by=actor.user_id,
                )
            )
    await flush_or_conflict(db)
    await db.refresh(item)
    item.options = (await _load_options_for_bank_items(db, [item.id])).get(item.id, [])
    return item


async def set_curated_bank_item_status(
    db: AsyncSession,
    *,
    course_id: UUID,
    item_id: UUID,
    status: str,
    actor: CurrentUser,
) -> QuizQuestionBankItem:
    if status not in {"approved", "archived"}:
        raise AppError("status must be approved or archived")
    item = (
        await db.execute(
            select(QuizQuestionBankItem).where(
                QuizQuestionBankItem.id == item_id,
                QuizQuestionBankItem.course_id == course_id,
                QuizQuestionBankItem.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Quiz question bank item {item_id} not found")
    if item.status == status:
        raise ConflictError(f"quiz_question_bank_item_already_{status}")
    if status == "approved":
        options = (await _load_options_for_bank_items(db, [item.id])).get(item.id, [])
        _validate_bank_content(
            _portable_question_content(item),
            [_portable_option_content(option) for option in options],
        )
        await _ensure_unique_bank_content(
            db,
            course_id=course_id,
            content_hash=item.content_hash,
            exclude_item_id=item.id,
        )
    item.status = status
    item.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(item)
    item.options = (await _load_options_for_bank_items(db, [item.id])).get(item.id, [])
    return item


async def delete_curated_bank_item(
    db: AsyncSession,
    *,
    course_id: UUID,
    item_id: UUID,
    actor: CurrentUser,
) -> None:
    item = (
        await db.execute(
            select(QuizQuestionBankItem).where(
                QuizQuestionBankItem.id == item_id,
                QuizQuestionBankItem.course_id == course_id,
                QuizQuestionBankItem.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Quiz question bank item {item_id} not found")
    await soft_delete_cascade(db, item, actor_id=actor.user_id)


async def import_curated_bank_items(
    db: AsyncSession,
    *,
    target_quiz_id: UUID,
    item_ids: list[UUID],
    actor: CurrentUser,
) -> list[QuizQuestion]:
    if len(item_ids) != len(set(item_ids)):
        raise AppError("item_ids contains duplicates")
    target = (await db.execute(select(Quiz).where(Quiz.id == target_quiz_id))).scalar_one_or_none()
    if target is None or target.deleted_at is not None:
        raise NotFoundError(f"Quiz {target_quiz_id} not found")
    _assert_target_editable(target)
    items = list(
        (
            await db.execute(
                select(QuizQuestionBankItem).where(
                    QuizQuestionBankItem.id.in_(item_ids),
                    QuizQuestionBankItem.course_id == target.course_id,
                    QuizQuestionBankItem.status == "approved",
                    QuizQuestionBankItem.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    if len(items) != len(item_ids):
        raise NotFoundError("One or more approved bank items were not found in course")
    by_id = {item.id: item for item in items}
    options_by_item = await _load_options_for_bank_items(db, item_ids)
    await _lock_question_append(db, target_quiz_id)
    position = await _next_position(db, target_quiz_id)
    created: list[QuizQuestion] = []
    for item_id in item_ids:
        item = by_id[item_id]
        created.append(
            await _clone_question_into_quiz(
                db,
                source=item,
                source_options=options_by_item.get(item.id, []),
                target_quiz_id=target_quiz_id,
                position=position,
                actor=actor,
                source_bank_item_id=item.id,
            )
        )
        position += 1
    hydrated = await _load_options_for_questions(db, [question.id for question in created])
    for question in created:
        question.options = hydrated.get(question.id, [])
    return created


__all__ = [
    "copy_questions_to_curated_bank",
    "create_curated_bank_item",
    "delete_curated_bank_item",
    "import_curated_bank_items",
    "list_curated_bank_items",
    "set_curated_bank_item_status",
    "update_curated_bank_item",
]
