"""Quiz persistence stage (T5.9).

Ports ``_persist_questions`` + ``_replace_question_in_place`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:871-958``. Both
helpers ``db.add()`` + ``db.flush()`` only; the caller owns the
transaction. ``QuizQuestion`` audit columns auto-populate via the T0.8
``before_flush`` listener; ``QuizQuestionRevision.created_by`` is set
explicitly because that table uses ``CreatedAtMixin`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from sqlalchemy import delete, func, select

from abridgeai.features.quizzes.models import (
    Quiz,
    QuizQuestion,
    QuizQuestionOption,
    QuizQuestionRevision,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.ai.retrieval import ChunkWithDistance


class _RunLike(Protocol):
    """Stub of ``GenerationRun`` — ``requested_by`` + ``config_json`` are read.

    ``config_json`` carries ``target_outcome_ids`` (§LO-3): when non-empty,
    every persisted question is round-robin bound to one of those outcomes.
    """

    requested_by: UUID | None
    config_json: dict[str, Any] | None


_VALID_FORMATS = frozenset({"plain", "markdown", "html"})


def _sanitize_rich_content(value: str | None, *, fmt: str) -> str | None:
    """Deferred import of ``services.sanitize``.

    A module-level import here is circular: importing the ``services``
    package eagerly imports ``generation`` -> ``ai.pipelines`` -> this
    package. Production never noticed (services always loads first), but
    any entry point that imports the persistence stage directly — tests,
    scripts — blew up with 'partially initialized module'.
    """
    from abridgeai.features.quizzes.services.sanitize import sanitize_rich_content

    return sanitize_rich_content(value, fmt=fmt)


def _fmt(raw: object) -> str:
    """Coerce a rich-format discriminator, failing safe to ``plain``.

    The parser already normalises these, but the persistence stage also runs on
    regeneration payloads assembled elsewhere, so re-validate at the write
    boundary rather than trusting the caller.
    """
    if isinstance(raw, str) and raw.strip().lower() in _VALID_FORMATS:
        return raw.strip().lower()
    return "plain"


def _apply_answer_fields(question: QuizQuestion, payload: dict[str, Any]) -> None:
    """Copy Phase 7 type-specific answer fields from a generated payload.

    Only fields present in ``payload`` are written, so a payload assembled by
    an older caller (or for a type that doesn't use them) leaves the column
    defaults intact. ``match_pairs``/``ordering_sequence`` land in JSONB
    columns, so they must be plain JSON — the parser already produces plain
    dicts/lists, and anything else is skipped rather than risking a
    serialization failure at flush time.
    """
    single_answer = payload.get("single_answer")
    if isinstance(single_answer, bool):
        question.single_answer = single_answer

    numeric_answer = payload.get("numeric_answer")
    if numeric_answer is not None:
        question.numeric_answer = numeric_answer

    numeric_tolerance = payload.get("numeric_tolerance")
    if numeric_tolerance is not None:
        question.numeric_tolerance = numeric_tolerance

    match_pairs = payload.get("match_pairs")
    if isinstance(match_pairs, list) and match_pairs:
        question.match_pairs = [
            {"left": str(pair.get("left", "")), "right": str(pair.get("right", ""))}
            for pair in match_pairs
            if isinstance(pair, dict)
        ]

    ordering_sequence = payload.get("ordering_sequence")
    if isinstance(ordering_sequence, list) and ordering_sequence:
        question.ordering_sequence = [str(item) for item in ordering_sequence]


def _structure_source_refs(
    raw_refs: object,
    chunks: list[ChunkWithDistance],
) -> list[dict[str, Any]]:
    """Hydrate generator chunk-id refs into JSONB ref dicts.

    Output schema: ``[{"chunk_id", "material_version_id", "course_id",
    "lesson_id"}]``. Order preserved; duplicates dropped; unknown ids
    skipped (mirrors legacy ``structure_source_refs``).
    """

    if not isinstance(raw_refs, list):
        return []

    chunks_by_id = {str(chunk.chunk_id): chunk for chunk in chunks}
    refs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_ref in raw_refs:
        chunk_id = _ref_chunk_id(raw_ref)
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        refs.append(
            {
                "chunk_id": str(chunk.chunk_id),
                "material_version_id": str(chunk.material_version_id),
                "course_id": str(chunk.course_id) if chunk.course_id else None,
                "lesson_id": str(chunk.lesson_id) if chunk.lesson_id else None,
            }
        )
    return refs


def _ref_chunk_id(raw_ref: object) -> str | None:
    if isinstance(raw_ref, str):
        return raw_ref
    if isinstance(raw_ref, dict):
        value = raw_ref.get("chunk_id") or raw_ref.get("id")
        return str(value) if value else None
    return None


def _parse_target_outcome_ids(config_json: dict[str, Any] | None) -> list[UUID]:
    """Read + coerce ``config_json['target_outcome_ids']`` to ``list[UUID]``.

    The service serializes them as strings; parse back defensively, dropping
    any malformed entry rather than aborting the whole run. Returns [] when
    the key is absent/empty (→ questions stay unassigned).
    """
    if not config_json:
        return []
    raw = config_json.get("target_outcome_ids")
    if not isinstance(raw, list):
        return []
    parsed: list[UUID] = []
    for item in raw:
        try:
            parsed.append(UUID(str(item)))
        except (ValueError, TypeError):
            continue
    return parsed


async def persist_questions(
    db: AsyncSession,
    run: _RunLike,
    quiz: Quiz,
    chunks: list[ChunkWithDistance],
    questions: list[dict[str, Any]],
) -> list[QuizQuestion]:
    """Create QuizQuestion + options + initial revision rows.

    Each ``questions`` entry: ``question_type``, ``prompt_text``,
    ``hint_text``, ``explanation``, ``difficulty``, ``bloom_level``,
    ``expected_response_time_ms``, ``source_refs`` (raw chunk-id list),
    ``original_generated_payload``, ``options`` (kwargs for
    ``QuizQuestionOption``). Positions append after the current max.
    """

    if not questions:
        return []

    position_result = await db.execute(
        select(func.coalesce(func.max(QuizQuestion.position), 0)).where(
            QuizQuestion.quiz_id == quiz.id
        )
    )
    start_position = int(position_result.scalar_one())

    # §LO-3 auto-assign: when the teacher targeted specific outcomes, bind
    # every generated question to one of them round-robin. This honours the
    # hard rule "if outcomes are given, no generated question is left
    # unassigned" deterministically, without depending on the LLM emitting a
    # correct outcome tag. Empty list → learning_outcome_id stays NULL and the
    # teacher assigns manually. Invalid UUID strings are skipped defensively.
    target_outcome_ids = _parse_target_outcome_ids(run.config_json)

    persisted: list[QuizQuestion] = []
    for offset, payload in enumerate(questions, start=1):
        structured_refs = _structure_source_refs(payload.get("source_refs"), chunks)
        learning_outcome_id = (
            target_outcome_ids[(offset - 1) % len(target_outcome_ids)]
            if target_outcome_ids
            else None
        )
        prompt_fmt = _fmt(payload.get("prompt_format"))
        hint_fmt = _fmt(payload.get("hint_format"))
        explanation_fmt = _fmt(payload.get("explanation_format"))
        question = QuizQuestion(
            quiz_id=quiz.id,
            position=start_position + offset,
            question_type=payload["question_type"],
            # Phase 3 SECURITY: AI content goes through the same nh3 cleaning as
            # the manual authoring path. Previously written raw — harmless while
            # everything was ``plain``, but stored XSS the moment a prompt emits
            # markdown/html, since the client renders those as HTML.
            prompt_text=_sanitize_rich_content(payload["prompt_text"], fmt=prompt_fmt),
            hint_text=_sanitize_rich_content(payload.get("hint_text"), fmt=hint_fmt),
            explanation=_sanitize_rich_content(
                payload.get("explanation"), fmt=explanation_fmt
            ),
            prompt_format=prompt_fmt,
            hint_format=hint_fmt,
            explanation_format=explanation_fmt,
            difficulty=payload.get("difficulty"),
            bloom_level=payload.get("bloom_level"),
            review_status="pending",
            expected_response_time_ms=payload.get("expected_response_time_ms"),
            learning_outcome_id=learning_outcome_id,
            source_refs=structured_refs,
            original_generated_payload=payload.get("original_generated_payload"),
        )
        # Phase 7: persist the type-specific answer key. Without this a
        # generated numerical/matching/ordering question would save with an
        # empty answer and be permanently ungradeable.
        _apply_answer_fields(question, payload)
        db.add(question)
        await db.flush()

        for option_payload in payload.get("options", []):
            db.add(QuizQuestionOption(question_id=question.id, **option_payload))

        db.add(
            QuizQuestionRevision(
                question_id=question.id,
                revision_no=1,
                source_kind="ai",
                payload_json=payload.get("original_generated_payload") or {},
                created_by=run.requested_by,
            )
        )
        await db.flush()
        persisted.append(question)

    return persisted


async def replace_question_in_place(
    db: AsyncSession,
    run: _RunLike,
    question: QuizQuestion,
    payload: dict[str, Any],
    chunks: list[ChunkWithDistance] | None = None,
) -> QuizQuestion:
    """Regenerate ``question`` in-place: same id, bumped ``revision_no``,
    review state reset to ``pending``, options atomically replaced.
    When ``chunks`` is provided, ``payload['source_refs']`` is re-hydrated.
    """

    structured_refs: list[dict[str, Any]] | Any = payload.get("source_refs")
    if chunks is not None:
        structured_refs = _structure_source_refs(structured_refs, chunks)

    prompt_fmt = _fmt(payload.get("prompt_format"))
    hint_fmt = _fmt(payload.get("hint_format"))
    explanation_fmt = _fmt(payload.get("explanation_format"))

    question.question_type = payload["question_type"]
    # Phase 3 SECURITY: same nh3 cleaning as the create path above.
    question.prompt_text = _sanitize_rich_content(payload["prompt_text"], fmt=prompt_fmt)
    question.hint_text = _sanitize_rich_content(payload.get("hint_text"), fmt=hint_fmt)
    question.explanation = _sanitize_rich_content(
        payload.get("explanation"), fmt=explanation_fmt
    )
    question.prompt_format = prompt_fmt
    question.hint_format = hint_fmt
    question.explanation_format = explanation_fmt
    # Phase 7: same answer-key copy as the create path.
    _apply_answer_fields(question, payload)
    question.difficulty = payload.get("difficulty")
    question.bloom_level = payload.get("bloom_level")
    question.review_status = "pending"
    question.expected_response_time_ms = payload.get("expected_response_time_ms")
    question.source_refs = structured_refs if structured_refs is not None else []
    question.original_generated_payload = payload.get("original_generated_payload")
    question.reviewed_by = None
    question.reviewed_at = None

    await db.execute(
        delete(QuizQuestionOption).where(QuizQuestionOption.question_id == question.id)
    )
    for option_payload in payload.get("options", []):
        db.add(QuizQuestionOption(question_id=question.id, **option_payload))

    revision_no_result = await db.execute(
        select(func.coalesce(func.max(QuizQuestionRevision.revision_no), 0)).where(
            QuizQuestionRevision.question_id == question.id
        )
    )
    db.add(
        QuizQuestionRevision(
            question_id=question.id,
            revision_no=int(revision_no_result.scalar_one()) + 1,
            source_kind="ai",
            payload_json=payload.get("original_generated_payload") or {},
            created_by=run.requested_by,
        )
    )
    await db.flush()
    return question


__all__ = ["persist_questions", "replace_question_in_place"]
