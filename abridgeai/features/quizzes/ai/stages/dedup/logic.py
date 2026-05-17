"""Quiz dedup stage (T5.8).

Ports ``_discard_duplicate_questions`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:1165-1234`` to the
feature-first layout, with the deliberate simplification mandated by
plan §5759: **deterministic exact match on normalized prompt + sorted
chunk_ids only**. The legacy Jaccard 5-gram near-dup layer is dropped
— teachers wanted predictable, explainable behaviour and false-positive
near-dup drops were a recurring complaint.

Cross-quiz collision detection delegates to
:func:`abridgeai.features.quizzes.queries.authoring.list_existing_module_question_keys`
(T5.3). That helper computes its hashes inside Postgres
(``sha256(prompt_text || md5(source_refs::text))``); to compare against
those hashes we *re-use the same SQL expression* for the candidate set
in a single round-trip rather than emulating Postgres' JSONB
canonicalisation in Python — JSONB key reordering and whitespace make
Python-side hashing fragile.

Drops carry human-readable ``reason`` strings (``EMPTY_PROMPT``,
``EXISTING_MODULE_DUPLICATE``, ``BATCH_DUPLICATE``) so the teacher-
review surface can render *why* a generated question was discarded
without callers parsing log lines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from abridgeai.features.quizzes.queries.authoring import (
    list_existing_module_question_keys,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.quizzes.models import Quiz


# Reason codes — kept short + machine-stable so callers can branch on
# them without locale-aware string parsing.
REASON_EMPTY_PROMPT = "EMPTY_PROMPT"
REASON_EXISTING_MODULE_DUPLICATE = "EXISTING_MODULE_DUPLICATE"
REASON_BATCH_DUPLICATE = "BATCH_DUPLICATE"


@dataclass(frozen=True)
class QuestionDrop:
    """A discarded question + why it was dropped.

    ``index`` is the candidate's 1-based position in the batch handed
    to :func:`discard_duplicates` (mirrors the legacy ``Q1: ...`` log
    format teachers see in audit trails). ``question`` is the original
    payload — kept verbatim so downstream UI can show the dropped text.
    """

    index: int
    reason: str
    question: dict[str, Any]


# Hash expression mirrors ``module_question_keys.sql`` exactly:
#
#     sha256( (prompt_text || md5(coalesce(source_refs::text, '[]'))) )
#
# We can't recompute that in Python because Postgres normalises JSONB
# (lexicographic key order, whitespace after ':' / ','). One round-trip
# with ``unnest`` keeps the cost flat regardless of batch size.
_CANDIDATE_KEYS_SQL = text(
    """
    SELECT
        candidate.idx AS idx,
        encode(
            sha256(
                (
                    candidate.prompt
                    || md5(coalesce(candidate.refs::jsonb::text, '[]'))
                )::bytea
            ),
            'hex'
        ) AS question_key
    FROM unnest(
        cast(:prompts as text[]),
        cast(:refs as text[])
    ) WITH ORDINALITY AS candidate(prompt, refs, idx)
    """
)


def _extract_source_refs(question: dict[str, Any]) -> list[Any]:
    """Pull the source-refs payload from a candidate question.

    Accepts both the legacy ``source_refs_json`` key (still emitted by
    parts of the haystack mapper layer) and the canonical
    ``source_refs`` key used by the new pipeline. Always returns a
    list — never ``None`` — so the SQL hash path never sees NULL.
    """

    refs = question.get("source_refs")
    if refs is None:
        refs = question.get("source_refs_json")
    if refs is None:
        return []
    if isinstance(refs, list):
        return refs
    # Defensive — generation could have returned a single dict / str
    return [refs]


async def _compute_candidate_keys(
    db: AsyncSession,
    candidates: list[tuple[str, list[Any]]],
) -> list[str]:
    """Compute the T5.3-compatible dedup hash for each candidate.

    Returns a list aligned with ``candidates`` (same length / order).
    Empty input short-circuits without a DB round-trip.
    """

    if not candidates:
        return []
    prompts = [prompt for prompt, _ in candidates]
    # Postgres needs each ``refs`` element as a JSON-encoded text so
    # the inline ``::jsonb::text`` cast can normalise it. ``[]`` is the
    # SQL default when the candidate emitted nothing.
    refs_text = [json.dumps(refs) if refs else "[]" for _, refs in candidates]
    rows = (
        await db.execute(
            _CANDIDATE_KEYS_SQL,
            {"prompts": prompts, "refs": refs_text},
        )
    ).all()
    # ``unnest WITH ORDINALITY`` is 1-based and unordered by default;
    # rebuild a stable list keyed by the index column.
    by_index: dict[int, str] = {row.idx: row.question_key for row in rows}
    return [by_index[i] for i in range(1, len(candidates) + 1)]


async def discard_duplicates(
    db: AsyncSession,
    quiz: Quiz,
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[QuestionDrop]]:
    """Strip duplicate questions out of a freshly-generated batch.

    Two collision layers, both deterministic:

    1. **Cross-quiz** — the candidate hash matches an existing question
       *anywhere in the same module* (per the T5.3 helper).
    2. **Intra-batch** — two candidates in the same batch hash to the
       same key (the LLM emitted a near-identical prompt twice).

    Empty prompts are dropped with reason ``EMPTY_PROMPT`` so an
    upstream parser bug surfaces in the drops list rather than
    poisoning the kept set with blank rows.

    Parameters
    ----------
    db
        Async session — used for both the T5.3 call and the candidate
        hash batch.
    quiz
        Quiz draft; only ``quiz.id`` is read but the parameter is the
        ORM row to keep the signature parallel to the other stages.
    questions
        Generated question payloads. Each must carry ``prompt_text``
        and (optionally) ``source_refs`` / ``source_refs_json``. Any
        other keys are passed through verbatim on the kept side and
        retained in :class:`QuestionDrop.question` on the dropped side.

    Returns
    -------
    tuple[list[dict[str, Any]], list[QuestionDrop]]
        ``(kept, drops)`` — ``kept`` preserves input order; ``drops``
        is in the order they were rejected so the teacher-review UI
        can render a deterministic timeline.
    """

    if not questions:
        return [], []

    existing_keys = await list_existing_module_question_keys(db, quiz.id)

    candidates: list[tuple[str, list[Any]]] = []
    structural_drops: list[QuestionDrop] = []
    candidate_indices: list[int] = []

    for i, question in enumerate(questions, start=1):
        prompt = (question.get("prompt_text") or "").strip()
        if not prompt:
            structural_drops.append(
                QuestionDrop(index=i, reason=REASON_EMPTY_PROMPT, question=question)
            )
            continue
        candidates.append((prompt, _extract_source_refs(question)))
        candidate_indices.append(i)

    keys = await _compute_candidate_keys(db, candidates)

    kept: list[dict[str, Any]] = []
    collision_drops: list[QuestionDrop] = []
    seen_in_batch: set[str] = set()

    for original_index, key, (_, _) in zip(candidate_indices, keys, candidates, strict=True):
        question = questions[original_index - 1]
        if key in existing_keys:
            collision_drops.append(
                QuestionDrop(
                    index=original_index,
                    reason=REASON_EXISTING_MODULE_DUPLICATE,
                    question=question,
                )
            )
            continue
        if key in seen_in_batch:
            collision_drops.append(
                QuestionDrop(
                    index=original_index,
                    reason=REASON_BATCH_DUPLICATE,
                    question=question,
                )
            )
            continue
        seen_in_batch.add(key)
        kept.append(question)

    # Merge drops in original-index order so callers see a stable
    # timeline regardless of which layer caught each collision.
    drops = sorted(structural_drops + collision_drops, key=lambda d: d.index)
    return kept, drops


__all__ = [
    "QuestionDrop",
    "REASON_BATCH_DUPLICATE",
    "REASON_EMPTY_PROMPT",
    "REASON_EXISTING_MODULE_DUPLICATE",
    "discard_duplicates",
]
