"""Course-scoped interview question bank authoring (§QBank-1).

Extracted from ``services.authoring`` (2026-09-01) to keep that module
under the interviews LOC ratchet. Owns the four-angle logical-group
model, the course bank CRUD, and the bank import-from-config flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.courses.api import public as courses_public
from abridgeai.features.interviews.dedup import store_question_embeddings
from abridgeai.features.interviews.models import (
    InterviewQuestion,
    InterviewQuestionBankItem,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.services import published_freeze
from abridgeai.features.interviews.services._shared import _require_config

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# --------------------------------------------------------------------------- #
# Course-scoped interview question bank (§QBank-1)
# --------------------------------------------------------------------------- #


async def _require_course(db: AsyncSession, course_id: UUID) -> None:
    """Ensure the course exists (permission is enforced at the router edge)."""
    course = await courses_public.get_course_by_id(db, course_id)
    if course is None:
        raise NotFoundError(f"Course {course_id} not found")


async def list_question_bank(db: AsyncSession, course_id: UUID) -> list[InterviewQuestionBankItem]:
    """All non-deleted bank items for a course, newest first."""
    from sqlalchemy import select  # noqa: PLC0415

    await _require_course(db, course_id)
    stmt = (
        select(InterviewQuestionBankItem)
        .where(
            InterviewQuestionBankItem.course_id == course_id,
            InterviewQuestionBankItem.deleted_at.is_(None),
        )
        .order_by(InterviewQuestionBankItem.created_at.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def add_to_question_bank(
    db: AsyncSession,
    course_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewQuestionBankItemCreate at edge
    actor: CurrentUser,
) -> InterviewQuestionBankItem:
    """Add a reusable question to the course bank (copy semantics)."""
    await _require_course(db, course_id)
    data = payload.model_dump(exclude_unset=True)
    item = await _create_bank_item(
        db,
        course_id=course_id,
        data=data,
        actor=actor,
        variant_group_id=None,
    )
    await flush_or_conflict(db)
    await db.refresh(item)
    return item


_LOGICAL_QUESTION_ANGLE_ORDER = (
    "technical",
    "system_design",
    "situational",
    "behavioral",
)
_LOGICAL_QUESTION_ANGLES = set(_LOGICAL_QUESTION_ANGLE_ORDER)
_LOGICAL_QUESTION_ANGLE_RANK = {
    angle: index for index, angle in enumerate(_LOGICAL_QUESTION_ANGLE_ORDER)
}


def _logical_angle_rank(item: InterviewQuestionBankItem) -> int:
    return _LOGICAL_QUESTION_ANGLE_RANK.get(item.question_type, len(_LOGICAL_QUESTION_ANGLE_ORDER))


def _normalise_prompt(value: str) -> str:
    return value.strip().lower()


async def _active_bank_group(
    db: AsyncSession, course_id: UUID, group_id: UUID
) -> list[InterviewQuestionBankItem]:
    from sqlalchemy import select  # noqa: PLC0415

    result = await db.execute(
        select(InterviewQuestionBankItem)
        .where(
            InterviewQuestionBankItem.course_id == course_id,
            InterviewQuestionBankItem.variant_group_id == group_id,
            InterviewQuestionBankItem.deleted_at.is_(None),
        )
        .order_by(InterviewQuestionBankItem.created_at)
    )
    # Canonical interviewer-angle order (created_at/id only break ties).
    return sorted(
        result.scalars().all(),
        key=lambda item: (_logical_angle_rank(item), item.created_at, item.id),
    )


async def _create_bank_item(
    db: AsyncSession,
    *,
    course_id: UUID,
    data: dict[str, Any],
    actor: CurrentUser,
    variant_group_id: UUID | None,
) -> InterviewQuestionBankItem:
    question_type = data["question_type"]
    if variant_group_id is not None and question_type not in _LOGICAL_QUESTION_ANGLES:
        raise AppError("logical_question_angle_required")
    prompt_text = str(data["prompt_text"] or "").strip()
    if not prompt_text:
        raise AppError("Question prompt is required")
    source_config_id = data.get("source_config_id")
    if source_config_id is not None:
        config = await _require_config(db, UUID(str(source_config_id)))
        if config.course_id != course_id:
            raise AppError("source_config_id does not belong to course")
    item = InterviewQuestionBankItem(
        course_id=course_id,
        prompt_text=prompt_text,
        question_type=question_type,
        difficulty=data.get("difficulty"),
        model_answer=(data.get("model_answer") or "").strip() or None,
        variant_group_id=variant_group_id,
        source_config_id=source_config_id,
        created_by=actor.user_id,
    )
    db.add(item)
    return item


async def create_question_bank_logical_group(
    db: AsyncSession,
    course_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> list[InterviewQuestionBankItem]:
    """Copy one complete four-angle logical question into the course bank."""
    await _require_course(db, course_id)
    payload_items = list(payload.items)
    prompts = [_normalise_prompt(str(item.prompt_text or "")) for item in payload_items]
    if len(prompts) != len(set(prompts)):
        raise ConflictError("logical_question_prompt_duplicate")
    group_id = uuid4()
    items = [
        await _create_bank_item(
            db,
            course_id=course_id,
            data=item.model_dump(exclude_unset=True),
            actor=actor,
            variant_group_id=group_id,
        )
        for item in payload.items
    ]
    await flush_or_conflict(db)
    for item in items:
        await db.refresh(item)
    return items


async def add_question_bank_sibling(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> list[InterviewQuestionBankItem]:
    """Append one missing angle to a singleton or partial logical bank group."""
    from sqlalchemy import select, update  # noqa: PLC0415

    anchor = (
        await db.execute(
            select(InterviewQuestionBankItem)
            .where(
                InterviewQuestionBankItem.id == item_id,
                InterviewQuestionBankItem.course_id == course_id,
                InterviewQuestionBankItem.deleted_at.is_(None),
            )
            .with_for_update()
        )
    ).scalar_one_or_none()
    if anchor is None:
        raise NotFoundError(f"Question bank item {item_id} not found")

    group_id = anchor.variant_group_id or uuid4()
    # Different children of an existing group are distinct rows, so their row
    # locks alone cannot serialize a same-angle sibling add. Coordinate writes
    # by the durable group id; singleton promotion is already serialized by the
    # anchor row lock above.
    if anchor.variant_group_id is not None:
        await authoring_queries.lock_bank_group(db, course_id, group_id)
    siblings = (
        await _active_bank_group(db, course_id, group_id) if anchor.variant_group_id else [anchor]
    )
    question_type = payload.question_type
    prompt_text = _normalise_prompt(str(payload.prompt_text or ""))
    if prompt_text in {_normalise_prompt(item.prompt_text) for item in siblings}:
        raise ConflictError("logical_question_prompt_duplicate")
    if question_type in {item.question_type for item in siblings}:
        raise ConflictError("logical_question_angle_already_present")
    if len(siblings) >= 4:
        raise ConflictError("logical_question_group_is_full")

    if anchor.variant_group_id is None:
        await db.execute(
            update(InterviewQuestionBankItem)
            .where(InterviewQuestionBankItem.id == anchor.id)
            .values(variant_group_id=group_id)
        )
        anchor.variant_group_id = group_id
    await _create_bank_item(
        db,
        course_id=course_id,
        data=payload.model_dump(exclude_unset=True),
        actor=actor,
        variant_group_id=group_id,
    )
    await flush_or_conflict(db)
    return await _active_bank_group(db, course_id, group_id)


def _order_bank_import(
    *,
    item_ids: list[UUID],
    selected: list[InterviewQuestionBankItem],
    expanded: list[InterviewQuestionBankItem],
) -> list[InterviewQuestionBankItem]:
    """Flatten the selection into insert order.

    Preserves the user's selected unit order, but always persists members of a
    logical question in its canonical interviewer-angle sequence. SQL and UUID
    order are intentionally never part of the authoring experience.
    """
    selected_by_id = {item.id: item for item in selected}
    unit_keys: list[tuple[str, UUID]] = []
    seen_units: set[tuple[str, UUID]] = set()
    for item_id in item_ids:
        item = selected_by_id[item_id]
        key = (
            ("group", item.variant_group_id)
            if item.variant_group_id is not None
            else ("item", item.id)
        )
        if key not in seen_units:
            seen_units.add(key)
            unit_keys.append(key)

    expanded_by_group: dict[UUID, list[InterviewQuestionBankItem]] = {}
    expanded_by_id = {item.id: item for item in expanded}
    for item in expanded:
        if item.variant_group_id is not None:
            expanded_by_group.setdefault(item.variant_group_id, []).append(item)

    ordered: list[InterviewQuestionBankItem] = []
    for kind_unit, unit_id in unit_keys:
        if kind_unit == "group":
            ordered.extend(
                sorted(
                    expanded_by_group[unit_id],
                    key=lambda item: (_logical_angle_rank(item), item.created_at, item.id),
                )
            )
        else:
            ordered.append(expanded_by_id[unit_id])
    return ordered


async def import_question_bank_items(
    db: AsyncSession,
    config_id: UUID,
    item_ids: list[UUID],
    actor: CurrentUser,
) -> list[InterviewQuestion]:
    """Atomically append chosen bank items, expanding logical siblings first."""
    from sqlalchemy import select  # noqa: PLC0415

    config = await _require_config(db, config_id)
    published_freeze.assert_questions_editable(config)
    if len(item_ids) != len(set(item_ids)):
        raise AppError("question_bank_import_item_ids contains duplicates")
    await authoring_queries.lock_question_append(db, config_id)
    selected = list(
        (
            await db.execute(
                select(InterviewQuestionBankItem).where(
                    InterviewQuestionBankItem.id.in_(set(item_ids)),
                    InterviewQuestionBankItem.course_id == config.course_id,
                    InterviewQuestionBankItem.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    if len(selected) != len(set(item_ids)):
        raise NotFoundError("One or more question-bank items were not found")

    group_ids = {item.variant_group_id for item in selected if item.variant_group_id is not None}
    expanded = list(selected)
    if group_ids:
        siblings = list(
            (
                await db.execute(
                    select(InterviewQuestionBankItem).where(
                        InterviewQuestionBankItem.course_id == config.course_id,
                        InterviewQuestionBankItem.variant_group_id.in_(group_ids),
                        InterviewQuestionBankItem.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        expanded = list({item.id: item for item in [*selected, *siblings]}.values())

    expanded_prompts = [_normalise_prompt(item.prompt_text) for item in expanded]
    if len(expanded_prompts) != len(set(expanded_prompts)):
        raise ConflictError("question_bank_import_prompt_duplicate")

    existing = {
        _normalise_prompt(question.prompt_text)
        for question in await authoring_queries.list_questions_for_config(db, config_id)
    }
    collisions = [item.id for item in expanded if _normalise_prompt(item.prompt_text) in existing]
    if collisions:
        raise ConflictError("question_bank_import_prompt_already_exists")

    # Preserve the user's selected unit order, but always persist members of a
    # logical question in its canonical interviewer-angle sequence.
    ordered = _order_bank_import(item_ids=item_ids, selected=selected, expanded=expanded)

    next_position = await authoring_queries.next_question_position(db, config_id)
    group_map = {group_id: uuid4() for group_id in group_ids}
    created: list[InterviewQuestion] = []
    for offset, item in enumerate(ordered):
        question = InterviewQuestion(
            interview_config_id=config_id,
            position=next_position + offset,
            question_type=item.question_type,
            prompt_text=item.prompt_text,
            difficulty=item.difficulty,
            model_answer=item.model_answer,
            review_status="approved",
            ai_generated=False,
            source_refs_json=[],
            source_module_ids=[],
            variant_group_id=(
                group_map[item.variant_group_id] if item.variant_group_id is not None else None
            ),
            reviewed_by=actor.user_id,
            reviewed_at=utcnow(),
            created_by=actor.user_id,
        )
        db.add(question)
        created.append(question)
    await flush_or_conflict(db)
    for question in created:
        await db.refresh(question)
    await store_question_embeddings(
        db,
        question_ids=[question.id for question in created],
        prompt_texts=[question.prompt_text for question in created],
    )
    return created


async def _assert_grouped_edit_keeps_group_valid(
    db: AsyncSession,
    *,
    course_id: UUID,
    item: InterviewQuestionBankItem,
    final_type: str | None,
    final_prompt: str,
) -> None:
    """Guard an edit of a logical-group member against its live siblings.

    Holds the group advisory lock first so two concurrent edits of different
    children cannot both pass the check and land the same angle. Errors are
    domain errors (400 / 409) — never an IntegrityError surfacing as a 500.
    """
    group_id = item.variant_group_id
    assert group_id is not None  # noqa: S101 -- caller checks; keeps mypy narrow
    await authoring_queries.lock_bank_group(db, course_id, group_id)
    siblings = [
        sibling
        for sibling in await _active_bank_group(db, course_id, group_id)
        if sibling.id != item.id
    ]
    if final_type not in _LOGICAL_QUESTION_ANGLES:
        raise AppError("logical_question_angle_required")
    if final_type in {sibling.question_type for sibling in siblings}:
        raise ConflictError("logical_question_angle_already_present")
    if _normalise_prompt(final_prompt) in {
        _normalise_prompt(sibling.prompt_text) for sibling in siblings
    }:
        raise ConflictError("logical_question_prompt_duplicate")


async def update_question_bank_item(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewQuestionBankItemUpdate at edge
    actor: CurrentUser,
) -> InterviewQuestionBankItem:
    """Edit one bank item without breaking its logical-question siblings."""
    from sqlalchemy import select  # noqa: PLC0415

    stmt = (
        select(InterviewQuestionBankItem)
        .where(
            InterviewQuestionBankItem.id == item_id,
            InterviewQuestionBankItem.course_id == course_id,
            InterviewQuestionBankItem.deleted_at.is_(None),
        )
        .with_for_update()
    )
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Question bank item {item_id} not found")

    data = payload.model_dump(exclude_unset=True)
    final_prompt = str(data.get("prompt_text", item.prompt_text) or "").strip()
    if not final_prompt:
        raise AppError("Question prompt is required")
    if "prompt_text" in data:
        data["prompt_text"] = final_prompt
    final_type = data.get("question_type", item.question_type)
    if final_type is None:
        # NOT NULL column: an explicit null is a bad request, not a DB error.
        raise AppError("question_type cannot be null")

    if item.variant_group_id is not None:
        await _assert_grouped_edit_keeps_group_valid(
            db,
            course_id=course_id,
            item=item,
            final_type=final_type,
            final_prompt=final_prompt,
        )

    for key, value in data.items():
        setattr(item, key, value)
    item.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(item)
    return item


async def delete_question_bank_item(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    actor: CurrentUser,
) -> None:
    """Soft-delete a bank item. Already-imported questions are untouched."""
    from sqlalchemy import select  # noqa: PLC0415

    stmt = select(InterviewQuestionBankItem).where(
        InterviewQuestionBankItem.id == item_id,
        InterviewQuestionBankItem.course_id == course_id,
        InterviewQuestionBankItem.deleted_at.is_(None),
    )
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Question bank item {item_id} not found")
    await soft_delete_cascade(db, item, actor_id=actor.user_id)


async def delete_question_bank_group(
    db: AsyncSession,
    course_id: UUID,
    item_id: UUID,
    actor: CurrentUser,
) -> int:
    """Soft-delete every live member of one logical question in the bank.

    Mirrors ``delete_question_variants`` on the config side: the named item
    establishes course scope, a singleton stays a one-row delete so the UI can
    expose one grouped action without branching, and grouped members are taken
    under the group advisory lock so a concurrent sibling-add cannot slip a new
    angle in behind the delete.
    """
    from sqlalchemy import select  # noqa: PLC0415

    stmt = select(InterviewQuestionBankItem).where(
        InterviewQuestionBankItem.id == item_id,
        InterviewQuestionBankItem.course_id == course_id,
        InterviewQuestionBankItem.deleted_at.is_(None),
    )
    item = (await db.execute(stmt)).scalar_one_or_none()
    if item is None:
        raise NotFoundError(f"Question bank item {item_id} not found")
    if item.variant_group_id is None:
        await soft_delete_cascade(db, item, actor_id=actor.user_id)
        return 1

    await authoring_queries.lock_bank_group(db, course_id, item.variant_group_id)
    members = await _active_bank_group(db, course_id, item.variant_group_id)
    for member in members:
        await soft_delete_cascade(db, member, actor_id=actor.user_id)
    return len(members)
