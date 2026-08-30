"""Question persistence for the interview generation pipeline.

Extracted from :mod:`.generation` (2026-08-29) to keep that orchestrator
under its LOC ratchet — mirroring the earlier ``variant.py`` extraction.
Owns draft → :class:`InterviewQuestion` row mapping, position allocation,
module attribution, and the best-effort batch embedding call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from abridgeai.core.observability import get_logger
from abridgeai.features.interviews.dedup import store_question_embeddings
from abridgeai.features.interviews.models import InterviewQuestion
from abridgeai.features.interviews.queries.authoring import (
    lock_question_append,
    next_question_position,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.generation.parsers import (
        InterviewQuestionDraft,
    )
    from abridgeai.features.interviews.models import InterviewConfig

logger = get_logger(__name__)

_DIFFICULTY_DRAFT_TO_ORM: dict[str, str] = {
    "easy": "junior",
    "medium": "mid_level",
    "hard": "senior",
}


def _persist_difficulty(value: str | None) -> str | None:
    if value is None:
        return None
    return _DIFFICULTY_DRAFT_TO_ORM.get(value, value)


def _module_ids_for_questions(
    config_json: dict[str, Any], config: InterviewConfig
) -> list[str]:
    """Module attribution for generated questions.

    Prefers the run's ``source_module_ids`` (the modules the teacher scoped
    generation to). Falls back to the interview config's own module so a
    question is never left unattributed.
    """
    raw = config_json.get("source_module_ids") or []
    ids = [str(m) for m in raw if m]
    if ids:
        return ids
    return [str(config.module_id)] if config.module_id is not None else []


async def _persist_questions(
    db: AsyncSession,
    *,
    config: InterviewConfig,
    accepted: list[InterviewQuestionDraft],
    source_module_ids: list[str],
    pipeline_run_id: UUID | None = None,
) -> None:
    await lock_question_append(db, config.id)
    next_position = await next_question_position(db, config.id)
    created: list[InterviewQuestion] = []
    for offset, draft in enumerate(accepted):
        position = next_position + offset
        question = InterviewQuestion(
            interview_config_id=config.id,
            linked_outcome_id=draft.linked_outcome_id,
            variant_group_id=draft.variant_group_id,
            position=position,
            question_type=draft.question_type,
            prompt_text=draft.prompt_text,
            difficulty=_persist_difficulty(draft.difficulty),
            model_answer=draft.model_answer.strip() or None,
            review_status="pending",
            ai_generated=True,
            source_refs_json=[str(c) for c in draft.source_refs],
            source_module_ids=source_module_ids,
        )
        db.add(question)
        await db.flush()
        created.append(question)

    # Embed the whole batch in ONE provider call, after the rows exist.
    #
    # This is the seam that was missing: duplicate detection only ever ran for
    # hand-authored questions, because `add_question` / `update_question` embed
    # but this pipeline did not. Since the shortlist skips `embedding IS NULL`
    # rows, an AI-generated bank was invisible to the checker and every
    # check-duplicate call on it answered "not a duplicate" — the feature looked
    # enabled and silently did nothing.
    #
    # Best-effort and already gated on `interview_dedup_enabled` inside the
    # helper: a provider failure here must not fail a completed generation run,
    # whose questions are otherwise perfectly valid.
    stored = await store_question_embeddings(
        db,
        question_ids=[q.id for q in created],
        prompt_texts=[q.prompt_text for q in created],
        pipeline_run_id=pipeline_run_id,
    )
    if stored:
        logger.info(
            "interview_generation_embedded",
            embedded=stored,
            total=len(created),
            config_id=str(config.id),
        )


__all__ = [
    "_module_ids_for_questions",
    "_persist_difficulty",
    "_persist_questions",
]
