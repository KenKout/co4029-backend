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

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.core.config import get_settings
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
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.gateway import LLMGateway


logger = logging.getLogger(__name__)


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


def _format_vector(embedding: list[float]) -> str:
    """Serialize to pgvector's text input format ``[a,b,c]``."""
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


async def store_question_embeddings(
    db: AsyncSession,
    *,
    question_ids: list[UUID],
    prompt_texts: list[str],
    embedding_client: EmbeddingClient | None = None,
    pipeline_run_id: UUID | None = None,
) -> int:
    """Embed and persist vectors for questions already flushed to the DB.

    Returns the number of rows given an embedding (0 when the feature is off,
    the input is empty, or the provider failed).

    Batches every prompt into ONE provider call. The alternative — embedding per
    question as it is created — costs one round trip per question, which is why
    the generation pipeline persisted whole runs with ``embedding = NULL``: nobody
    wanted N sequential provider calls inside a run.

    Best-effort by contract. A missing vector degrades duplicate detection to
    "no candidates found" (``shortlist_similar_questions`` skips NULL rows), so
    losing one must never fail a teacher's save or abort a generation run. All
    failures are logged and swallowed.

    Skipped entirely when ``interview_dedup_enabled`` is off — no embedding spend
    for a feature nobody turned on.
    """
    if not get_settings().interview_dedup_enabled:
        return 0
    if len(question_ids) != len(prompt_texts):
        raise ValueError("question_ids and prompt_texts must be the same length")

    # Blank prompts can't be embedded; drop them before paying for a call.
    pairs = [
        (qid, text_value.strip())
        for qid, text_value in zip(question_ids, prompt_texts, strict=True)
        if (text_value or "").strip()
    ]
    if not pairs:
        return 0

    try:
        client = embedding_client or EmbeddingClient()
        vectors = await client.embed(
            [prompt for _, prompt in pairs],
            db=db,
            pipeline_run_id=pipeline_run_id,
        )
    except Exception:  # noqa: BLE001 — see docstring; never break the caller
        logger.warning(
            "interview dedup: failed to embed %d question(s); leaving embedding NULL",
            len(pairs),
            exc_info=True,
        )
        return 0

    stored = 0
    for (question_id, _prompt), vector in zip(pairs, vectors, strict=False):
        if not vector:
            continue
        try:
            await db.execute(
                text(
                    "UPDATE interview_questions "
                    "SET embedding = CAST(:embedding AS halfvec) "
                    "WHERE id = :question_id"
                ),
                {
                    "embedding": _format_vector(vector),
                    "question_id": question_id,
                },
            )
            stored += 1
        except Exception:  # noqa: BLE001 — one bad row must not lose the rest
            logger.warning(
                "interview dedup: failed to store embedding for question %s",
                question_id,
                exc_info=True,
            )
    return stored


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
    "store_question_embeddings",
]
