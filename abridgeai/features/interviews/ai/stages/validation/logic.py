"""Interview validation stage orchestrator (T6.6).

Combines four deterministic Python checks (GROUNDED,
DIFFICULTY_COHERENT, TYPE_MATCHES_CONFIG, LENGTH_REASONABLE) with one
LLM-judged check (NOT_LEADING) into a positional list of
:class:`Verdict` objects parallel to the input drafts.

Audit fields
------------
* ``stage_name="interview_validation"`` — required so
  ``ai_model_calls`` rows roll up to this stage in the run.
* ``pipeline_run_id`` — threaded into the gateway so per-call cost
  rolls up to the parent ``ai_pipeline_runs`` row.

The ``InterviewRetrievalContext`` Protocol declared here is a
placeholder until T6.4 lands its concrete dataclass. Any retrieval
stage producing an object with a ``chunks`` attribute (whose elements
expose an ``id`` UUID) satisfies the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.generation.resolve import (
    VARIANT_ANGLES,
)
from abridgeai.features.interviews.ai.stages.validation.parsers import (
    parse_leading_verdicts,
)
from abridgeai.features.interviews.ai.stages.validation.verdicts import (
    ValidationCriterion,
    Verdict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


VALIDATION_STAGE_NAME = "interview_validation"

DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "technical": 0.60,
    "behavioral": 0.30,
    "situational": 0.10,
}
TYPE_WEIGHT_TOLERANCE = 0.10
TYPE_MIX_MIN_BATCH = 5
MIN_PROMPT_CHARS = 20
MAX_PROMPT_CHARS = 500
_DIFFICULTY_RANK: dict[str, int] = {"easy": 1, "medium": 2, "hard": 3}


class _ChunkLike(Protocol):
    """Minimal contract for the chunk objects produced by retrieval."""

    id: UUID


class InterviewRetrievalContext(Protocol):
    """Minimal contract for the retrieval stage output (T6.4)."""

    chunks: Sequence[_ChunkLike]


class InterviewQuestionDraft(Protocol):
    """Minimal contract for the generation stage output (T6.5).

    Mirrors the dataclass fields produced by
    :mod:`abridgeai.features.interviews.ai.stages.generation.parsers`.
    Declared as a Protocol so the validation stage stays decoupled —
    any object with these attributes (real draft, test stub, future
    refactor) can be validated.
    """

    question_type: str
    prompt_text: str
    difficulty: str
    expected_depth: int
    source_refs: Sequence[UUID]
    linked_outcome_id: UUID | None
    variant_group_id: UUID | None


class GenerationRun(Protocol):
    """Minimal contract for the run row owning the validation call.

    Cross-feature imports of ``abridgeai.features.quizzes.models`` are
    forbidden by the importlinter independence contract, so we declare
    the shape inline. Any object exposing ``id`` (UUID FK target on
    ``ai_pipeline_runs``) and a ``config_json`` dict satisfies it.
    """

    id: UUID
    config_json: dict[str, Any]


class InterviewConfig(Protocol):
    """Minimal contract for the interview config row.

    Same rationale as :class:`GenerationRun` — declared inline to keep
    the validation stage usable without a hard import dependency on
    ``abridgeai.features.interviews.models.InterviewConfig``.
    """

    title: str
    persona: str | None


async def validate_interview_questions(
    db: AsyncSession,
    *,
    run: GenerationRun,
    config: InterviewConfig,
    drafts: list[InterviewQuestionDraft],
    context: InterviewRetrievalContext,
    gateway: LLMGateway | None = None,
    skip_type_mix: bool = False,
) -> list[Verdict]:
    """Validate ``drafts`` and return a positional list of verdicts.

    Each draft yields exactly one :class:`Verdict`. ``failed_criteria``
    accumulates *every* check that failed, so the caller (T6.10) can
    decide whether to drop, regenerate, or surface the question for
    teacher review.
    """

    if not drafts:
        return []

    deterministic = _run_deterministic_checks(
        drafts, run=run, context=context, skip_type_mix=skip_type_mix
    )
    leading = await _run_leading_check(
        drafts=drafts,
        config=config,
        run=run,
        db=db,
        gateway=gateway,
    )

    verdicts: list[Verdict] = []
    for index, draft in enumerate(drafts):
        failed = list(deterministic[index])
        if not leading[index]:
            failed.append(ValidationCriterion.NOT_LEADING)
        rationale = _build_rationale(draft, failed)
        verdicts.append(
            Verdict(
                question_index=index,
                accepted=not failed,
                failed_criteria=failed,
                rationale=rationale,
            )
        )
    return verdicts


def _run_deterministic_checks(
    drafts: list[InterviewQuestionDraft],
    *,
    run: GenerationRun,
    context: InterviewRetrievalContext,
    skip_type_mix: bool = False,
) -> list[list[ValidationCriterion]]:
    """Apply the four Python-only checks; return failures per question.

    ``skip_type_mix`` is set for variant-mode generation (``all_angles`` /
    ``role_only``), where the type distribution is an explicit per-angle
    schedule rather than the 60/30/10 mix the check enforces.
    """
    chunk_ids = _collect_chunk_ids(context)
    type_failures: set[int] = set()
    if not skip_type_mix:
        type_failures = _check_type_mix(drafts, _resolve_type_weights(run))
    group_failures = _check_variant_groups(drafts)
    failures: list[list[ValidationCriterion]] = []
    for index, draft in enumerate(drafts):
        per_q: list[ValidationCriterion] = []
        if not _is_grounded(draft, chunk_ids):
            per_q.append(ValidationCriterion.GROUNDED)
        if not _has_coherent_difficulty(drafts, index):
            per_q.append(ValidationCriterion.DIFFICULTY_COHERENT)
        if index in type_failures:
            per_q.append(ValidationCriterion.TYPE_MATCHES_CONFIG)
        if not _has_reasonable_length(draft):
            per_q.append(ValidationCriterion.LENGTH_REASONABLE)
        if index in group_failures:
            per_q.append(ValidationCriterion.VARIANT_GROUP_COHERENT)
        failures.append(per_q)
    return failures


def _check_variant_groups(drafts: list[InterviewQuestionDraft]) -> set[int]:
    """Reject structurally inconsistent all-angle groups, allow partial groups."""
    groups: dict[UUID, list[tuple[int, InterviewQuestionDraft]]] = {}
    for index, draft in enumerate(drafts):
        group_id = getattr(draft, "variant_group_id", None)
        if isinstance(group_id, UUID):
            groups.setdefault(group_id, []).append((index, draft))

    failed: set[int] = set()
    for members in groups.values():
        outcomes = {member.linked_outcome_id for _, member in members}
        difficulties = {member.difficulty for _, member in members}
        types = {member.question_type for _, member in members}
        valid = (
            len(members) == len(VARIANT_ANGLES)
            and types == set(VARIANT_ANGLES)
            and len(outcomes) == 1
            and len(difficulties) == 1
        )
        if not valid:
            failed.update(index for index, _ in members)
    return failed


def _collect_chunk_ids(context: InterviewRetrievalContext) -> set[UUID]:
    out: set[UUID] = set()
    for chunk in getattr(context, "chunks", []) or []:
        chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)
        if isinstance(chunk_id, UUID):
            out.add(chunk_id)
    return out


def _is_grounded(draft: InterviewQuestionDraft, chunk_ids: set[UUID]) -> bool:
    if not draft.source_refs:
        return False
    return any(ref in chunk_ids for ref in draft.source_refs)


def _has_coherent_difficulty(drafts: list[InterviewQuestionDraft], index: int) -> bool:
    """A question is incoherent when difficulty drops by more than one
    rank from the previous question (e.g. hard → easy)."""
    if index == 0:
        return True
    prev = _DIFFICULTY_RANK.get(drafts[index - 1].difficulty, 2)
    curr = _DIFFICULTY_RANK.get(drafts[index].difficulty, 2)
    return curr >= prev - 1


def _has_reasonable_length(draft: InterviewQuestionDraft) -> bool:
    length = len(draft.prompt_text)
    return MIN_PROMPT_CHARS <= length <= MAX_PROMPT_CHARS


def _resolve_type_weights(run: GenerationRun) -> dict[str, float]:
    config_json = getattr(run, "config_json", None) or {}
    raw = config_json.get("type_weights") if isinstance(config_json, dict) else None
    if not isinstance(raw, dict):
        return dict(DEFAULT_TYPE_WEIGHTS)
    cleaned: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue
    return cleaned or dict(DEFAULT_TYPE_WEIGHTS)


def _check_type_mix(
    drafts: list[InterviewQuestionDraft],
    weights: dict[str, float],
) -> set[int]:
    """Return indices belonging to overrepresented type buckets.

    A bucket is overrepresented when its observed share exceeds
    ``target + TYPE_WEIGHT_TOLERANCE``. Underrepresented buckets are
    not flagged on individual questions — there is nothing to remove.
    Below ``TYPE_MIX_MIN_BATCH`` the check is skipped: with only a
    handful of drafts the type-share ratios are noisy and the plan's
    8-12 question target hasn't kicked in yet (an outcome-based
    interview can legitimately have a single question).
    """
    if len(drafts) < TYPE_MIX_MIN_BATCH:
        return set()
    counts: dict[str, int] = {}
    for draft in drafts:
        counts[draft.question_type] = counts.get(draft.question_type, 0) + 1
    total = len(drafts)
    overrepresented: set[str] = set()
    for qtype, count in counts.items():
        target = weights.get(qtype, 0.0)
        observed = count / total
        if observed - target > TYPE_WEIGHT_TOLERANCE:
            overrepresented.add(qtype)
    if not overrepresented:
        return set()
    return {index for index, draft in enumerate(drafts) if draft.question_type in overrepresented}


async def _run_leading_check(
    *,
    drafts: list[InterviewQuestionDraft],
    config: InterviewConfig,
    run: GenerationRun,
    db: AsyncSession,
    gateway: LLMGateway | None,
) -> list[bool]:
    """Ask the LLM to judge each question's neutrality."""
    gateway = gateway or LLMGateway()
    chunk_views = _chunk_views_for_prompt(run)
    questions = [
        {
            "index": index,
            "question_type": draft.question_type,
            "difficulty": draft.difficulty,
            "expected_depth": draft.expected_depth,
            "prompt_text": draft.prompt_text,
        }
        for index, draft in enumerate(drafts)
    ]
    config_summary = _config_summary(config)
    system_prompt = render_prompt("prompts/system.j2")
    user_prompt = render_prompt(
        "prompts/user.j2",
        title=getattr(config, "title", ""),
        config_summary=config_summary,
        chunks=chunk_views,
        questions=questions,
    )
    llm_result = await gateway.generate_json(
        role=LLMRole.INTERVIEW_VALIDATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name=VALIDATION_STAGE_NAME,
        pipeline_run_id=getattr(run, "id", None),
        parent_run_id=getattr(run, "id", None),
    )
    payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else {}
    parsed = parse_leading_verdicts(payload, question_count=len(drafts))
    return [verdict.not_leading for verdict in parsed]


def _chunk_views_for_prompt(run: GenerationRun) -> list[dict[str, Any]]:
    config_json = getattr(run, "config_json", None) or {}
    retrieval = config_json.get("retrieval") if isinstance(config_json, dict) else None
    if not isinstance(retrieval, dict):
        return []
    raw_ids = retrieval.get("source_chunk_ids")
    if not isinstance(raw_ids, list):
        return []
    return [{"id": str(chunk_id), "content": ""} for chunk_id in raw_ids if chunk_id]


def _config_summary(config: InterviewConfig) -> str:
    parts: list[str] = []
    persona = getattr(config, "persona", None)
    if persona:
        parts.append(f"persona={persona}")
    return ", ".join(parts)


def _build_rationale(
    draft: InterviewQuestionDraft,
    failed: Iterable[ValidationCriterion],
) -> str:
    failed_list = list(failed)
    if not failed_list:
        return f"Accepted: '{draft.prompt_text[:60]}'"
    codes = ", ".join(criterion.value for criterion in failed_list)
    return f"Failed criteria: {codes}"


__all__ = [
    "DEFAULT_TYPE_WEIGHTS",
    "InterviewRetrievalContext",
    "MAX_PROMPT_CHARS",
    "MIN_PROMPT_CHARS",
    "TYPE_WEIGHT_TOLERANCE",
    "VALIDATION_STAGE_NAME",
    "validate_interview_questions",
]
