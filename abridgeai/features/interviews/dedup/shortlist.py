"""Two-stage duplicate detection for the interview question bank.

Stage 1 (this module, cheap): embed the candidate question and pull the nearest
existing questions in the same config by pgvector cosine distance. This only
*shortlists* — it never decides.

Stage 2 (``judge.py``, one LLM call): decide whether the candidate is genuinely
asking the same thing as any shortlisted question.

Why two stages rather than a similarity threshold alone
-------------------------------------------------------
Cosine similarity cannot tell "same question, reworded" from "same topic,
different angle", and in an exam bank those have opposite handling:

* "What challenges did you hit on project X?" vs "Tell me about difficulties in
  project X" — duplicate, drop one.
* "What was your routine in college?" vs "...in your first job?" — NOT a
  duplicate; different period, different answer.

Both pairs land in the same similarity band, so any single cutoff is wrong for
one of them. The vector stage exists to keep the LLM's input to a handful of
plausible candidates instead of the whole bank; the semantic call makes the
judgement. Ported from SALT-NLP/SparkMe's question-bank dedup, adapted for
assessment (their bank is per-interviewee; ours is per-config and teacher-owned).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.llm.embeddings import EmbeddingClient

# How many nearest questions to hand the judge. Small on purpose: the judge sees
# every candidate in one prompt, and a long list both costs more and dilutes the
# model's attention across obviously-unrelated questions.
SHORTLIST_SIZE = 5

# Cosine distance above which a question is not even worth judging. Deliberately
# LOOSE (0.55 distance ≈ 0.45 similarity): the vector stage must not be the thing
# that decides, so it errs toward passing candidates through. Tightening this
# trades recall for LLM spend, and missing a duplicate is the worse error here.
MAX_SHORTLIST_DISTANCE = 0.55


@dataclass(frozen=True, slots=True)
class ShortlistedQuestion:
    """An existing bank question that is close enough to be worth judging."""

    question_id: UUID
    prompt_text: str
    distance: float
    """Raw pgvector cosine distance in ``[0.0, 2.0]``; smaller is more similar."""

    @property
    def similarity(self) -> float:
        """Convenience score in ``[-1.0, 1.0]``; larger is more similar."""
        return 1.0 - self.distance


async def embed_question(
    db: AsyncSession,
    *,
    prompt_text: str,
    embedding_client: EmbeddingClient,
) -> list[float] | None:
    """Embed one question prompt, or return ``None`` if it cannot be embedded.

    Returns ``None`` for blank input rather than calling the provider. Any
    provider failure is left to propagate: the caller decides whether duplicate
    detection is optional for its flow (during authoring it is).
    """
    cleaned = (prompt_text or "").strip()
    if not cleaned:
        return None
    vectors = await embedding_client.embed([cleaned], db=db)
    return vectors[0] if vectors else None


async def shortlist_similar_questions(
    db: AsyncSession,
    *,
    config_id: UUID,
    embedding: list[float],
    exclude_question_id: UUID | None = None,
    limit: int = SHORTLIST_SIZE,
    max_distance: float = MAX_SHORTLIST_DISTANCE,
) -> list[ShortlistedQuestion]:
    """Nearest existing questions in ``config_id``, closest first.

    Scoped to a single config: a "duplicate" only means anything within one
    interview's bank, and two configs legitimately reuse the same question.

    Rows with a ``NULL`` embedding are skipped — not-yet-embedded is treated as
    "unknown", never as "dissimilar". Soft-deleted rows are excluded so a
    question the teacher already removed cannot block a new one.
    """
    if not embedding:
        return []

    # `AND (:exclude_id IS NULL OR id <> :exclude_id)` looks tidier but Postgres
    # cannot infer a type for a bare NULL parameter in that position and raises
    # AmbiguousParameter. Build the clause only when there is an id to exclude,
    # and keep the parameter out of the query entirely otherwise.
    params: dict[str, object] = {
        "embedding": _format_vector(embedding),
        "config_id": config_id,
        "max_distance": max_distance,
        "limit": limit,
    }
    exclude_clause = ""
    if exclude_question_id is not None:
        exclude_clause = "  AND id <> :exclude_id "
        params["exclude_id"] = exclude_question_id

    sql = text(
        "SELECT id, prompt_text, embedding <=> CAST(:embedding AS halfvec) AS distance "
        "FROM interview_questions "
        "WHERE interview_config_id = :config_id "
        "  AND embedding IS NOT NULL "
        "  AND deleted_at IS NULL "
        + exclude_clause
        + "  AND embedding <=> CAST(:embedding AS halfvec) <= :max_distance "
        "ORDER BY distance "
        "LIMIT :limit"
    )
    rows = (await db.execute(sql, params)).all()

    return [
        ShortlistedQuestion(
            question_id=row[0],
            prompt_text=row[1] or "",
            distance=float(row[2]),
        )
        for row in rows
    ]


def _format_vector(embedding: list[float]) -> str:
    """Serialize to pgvector's text input format ``[a,b,c]``.

    Mirrors ``abridgeai.ai.retrieval.pgvector._format_vector``; duplicated rather
    than imported because that module is scoped to ``document_chunks`` retrieval
    and its helper is private.
    """
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


__all__ = [
    "MAX_SHORTLIST_DISTANCE",
    "SHORTLIST_SIZE",
    "ShortlistedQuestion",
    "embed_question",
    "shortlist_similar_questions",
]
