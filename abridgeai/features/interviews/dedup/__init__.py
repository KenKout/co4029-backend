"""Two-stage duplicate detection for the interview question bank.

Public entry point is :func:`check_duplicate`, which runs both stages:

1. :mod:`shortlist` — embed the proposal, pull the nearest existing questions in
   the same config by pgvector cosine distance. Cheap; only narrows candidates.
2. :mod:`judge` — one LLM call decides whether it is genuinely the same ask.

A similarity threshold alone cannot separate "reworded duplicate" from "same topic,
different angle", and in an exam bank those need opposite handling — see
``shortlist.py`` for the worked examples. Behind ``interview_dedup_enabled``
(default off).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abridgeai.features.interviews.dedup.judge import (
    DEDUP_STAGE_NAME,
    NOT_DUPLICATE,
    DuplicateVerdict,
    judge_duplicate,
    parse_duplicate_verdict,
)
from abridgeai.features.interviews.dedup.shortlist import (
    MAX_SHORTLIST_DISTANCE,
    SHORTLIST_SIZE,
    ShortlistedQuestion,
    embed_question,
    shortlist_similar_questions,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.embeddings import EmbeddingClient
    from abridgeai.ai.llm.gateway import LLMGateway


async def check_duplicate(
    db: AsyncSession,
    *,
    config_id: UUID,
    prompt_text: str,
    embedding_client: EmbeddingClient,
    exclude_question_id: UUID | None = None,
    gateway: LLMGateway | None = None,
    pipeline_run_id: UUID | None = None,
) -> tuple[DuplicateVerdict, list[float] | None]:
    """Run both stages for one proposed question.

    Returns ``(verdict, embedding)``. The embedding is handed back so a caller
    that goes on to persist the question can store it without paying for a second
    embed — that vector is what makes the *next* question's check possible.

    Fails open at every step: an embedding failure, an empty shortlist, or a judge
    failure all yield a non-duplicate verdict (with ``error`` set where something
    actually broke), because blocking a teacher's save is worse than admitting one
    near-duplicate that they can still delete.
    """
    try:
        embedding = await embed_question(
            db, prompt_text=prompt_text, embedding_client=embedding_client
        )
    except Exception as exc:  # noqa: BLE001 — fail open; embedding is best-effort here
        return (
            DuplicateVerdict(is_duplicate=False, error=f"embedding failed: {exc}"),
            None,
        )

    if not embedding:
        return NOT_DUPLICATE, None

    candidates = await shortlist_similar_questions(
        db,
        config_id=config_id,
        embedding=embedding,
        exclude_question_id=exclude_question_id,
    )
    verdict = await judge_duplicate(
        db,
        prompt_text=prompt_text,
        candidates=candidates,
        gateway=gateway,
        pipeline_run_id=pipeline_run_id,
    )
    return verdict, embedding


__all__ = [
    "DEDUP_STAGE_NAME",
    "MAX_SHORTLIST_DISTANCE",
    "NOT_DUPLICATE",
    "SHORTLIST_SIZE",
    "DuplicateVerdict",
    "ShortlistedQuestion",
    "check_duplicate",
    "embed_question",
    "judge_duplicate",
    "parse_duplicate_verdict",
    "shortlist_similar_questions",
]
