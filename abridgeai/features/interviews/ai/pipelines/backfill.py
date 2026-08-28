"""Generate+validate backfill loop for the interview pipeline (T6.10).

Validation (T6.6) drops drafts that fail its checks, so one generation
call rarely lands exactly on the requested ``question_count``.
:func:`generate_with_backfill` re-runs generate+validate in rounds — each
asking for exactly the remaining shortfall and telling the LLM which
prompts are already accepted (``avoid_prompts``) — until the count is
met or a round adds nothing new, bounded by ``MAX_BACKFILL_ATTEMPTS``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from abridgeai.core.observability import get_logger
from abridgeai.features.interviews.ai.stages.generation import (
    generate_interview_questions,
)
from abridgeai.features.interviews.ai.stages.generation.resolve import VARIANT_ANGLES
from abridgeai.features.interviews.ai.stages.validation import (
    validate_interview_questions,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.ai.stages.generation.parsers import (
        InterviewQuestionDraft,
    )
    from abridgeai.features.interviews.ai.stages.retrieval.logic import (
        InterviewRetrievalContext,
    )
    from abridgeai.features.interviews.ai.stages.validation.verdicts import Verdict
    from abridgeai.features.interviews.models import InterviewConfig

# Bounds how many extra generate+validate rounds the loop below will run —
# protects LLM spend/latency on a source that genuinely cannot support the
# requested count (e.g. too few indexed chunks to ground more questions).
MAX_BACKFILL_ATTEMPTS = 3

logger = get_logger(__name__)


def accepted_drafts(
    drafts: list[InterviewQuestionDraft],
    verdicts: list[Verdict],
) -> list[InterviewQuestionDraft]:
    """Positional zip of drafts/verdicts, keeping only accepted drafts."""
    return [d for i, d in enumerate(drafts) if i < len(verdicts) and verdicts[i].accepted]


def validation_summary(verdicts: list[Verdict]) -> dict[str, Any]:
    rejected = [v for v in verdicts if not v.accepted]
    failure_codes: dict[str, int] = {}
    for verdict in rejected:
        for criterion in verdict.failed_criteria:
            failure_codes[criterion.value] = failure_codes.get(criterion.value, 0) + 1
    return {
        "accepted": sum(1 for v in verdicts if v.accepted),
        "rejected": len(rejected),
        "failures": failure_codes,
    }


def _accepted_complete_groups(
    drafts: list[InterviewQuestionDraft],
    verdicts: list[Verdict],
    seen_prompts: set[str],
) -> list[list[InterviewQuestionDraft]]:
    """Return only complete all-angle groups whose members all passed."""
    verdict_by_draft = {
        id(draft): verdict for draft, verdict in zip(drafts, verdicts, strict=False)
    }
    groups: dict[object, list[InterviewQuestionDraft]] = {}
    for draft in drafts:
        if draft.variant_group_id is not None:
            groups.setdefault(draft.variant_group_id, []).append(draft)

    accepted: list[list[InterviewQuestionDraft]] = []
    for members in groups.values():
        types = {draft.question_type for draft in members}
        if (
            len(members) != len(VARIANT_ANGLES)
            or types != set(VARIANT_ANGLES)
            or any(
                verdict_by_draft.get(id(draft)) is None
                or not verdict_by_draft[id(draft)].accepted
                or draft.prompt_text in seen_prompts
                for draft in members
            )
        ):
            continue
        by_type = {draft.question_type: draft for draft in members}
        accepted.append([by_type[angle] for angle in VARIANT_ANGLES])
    return accepted


def _trim_groups_to_shortfall(
    groups: list[list[InterviewQuestionDraft]],
    missing: int,
) -> list[InterviewQuestionDraft]:
    """Keep complete groups only; all-angle targets are always multiples of four."""
    group_capacity = missing // len(VARIANT_ANGLES)
    return [draft for group in groups[:group_capacity] for draft in group]


async def generate_with_backfill(
    db: AsyncSession,
    *,
    state: Any,  # noqa: ANN401 -- GenerationRunDTO duck-type, avoids quizzes import
    config: InterviewConfig,
    context: InterviewRetrievalContext,
    outcomes: list[Any],
    target_count: int,
    variant_strategy: str | None = None,
    role_type: str | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> tuple[list[InterviewQuestionDraft], list[Verdict], list[InterviewQuestionDraft], int]:
    """Generate+validate in rounds until ``target_count`` accepted drafts.

    De-dupes accepted drafts by ``prompt_text`` across rounds (belt-and-
    suspenders against a model that repeats a prompt despite
    ``avoid_prompts``). Stops early once a round adds nothing new.

    ``on_progress(accepted_so_far, target_count)`` — if supplied — is awaited
    after each round so the caller can persist live progress (the teacher UI
    polls ``generation_runs.config_json`` while the run is ``running``).
    """
    if variant_strategy == "all_angles" and target_count % len(VARIANT_ANGLES) != 0:
        raise ValueError("all_angles target_count must be divisible by the angle count")

    seen_prompts: set[str] = set()
    all_drafts: list[InterviewQuestionDraft] = []
    all_verdicts: list[Verdict] = []
    accepted: list[InterviewQuestionDraft] = []
    backfill_rounds = 0

    for attempt in range(1 + MAX_BACKFILL_ATTEMPTS):
        missing = target_count - len(accepted)
        if missing <= 0:
            break
        if attempt > 0:
            backfill_rounds += 1
        # Backfill rounds ask for a small buffer above the exact shortfall
        # so a typical rejection rate still lands on target without yet
        # another round.
        request_count = missing if attempt == 0 else min(missing + 1, missing * 2)

        # Variant mode (all_angles): ``override_question_count`` is a TOTAL row
        # budget and the generation stage re-divides it by the angle count
        # (ceil). ``accepted`` only ever grows by whole angle groups, so
        # ``missing`` is always a multiple of four here — ask for exactly that
        # many rows (no surplus buffer) so the run lands exactly on target.
        trim_to_total = False
        if variant_strategy == "all_angles":
            request_count = missing
            trim_to_total = True

        round_drafts = await generate_interview_questions(
            db,
            run=state,
            config=config,
            context=cast("Any", context),
            outcomes=cast("Any", outcomes),
            override_question_count=request_count,
            avoid_prompts=[d.prompt_text for d in accepted],
            variant_strategy=variant_strategy,
            role_type=role_type,
        )
        round_verdicts = await validate_interview_questions(
            db,
            run=state,
            config=config,
            drafts=cast("Any", round_drafts),
            context=cast("Any", context),
            skip_type_mix=variant_strategy is not None,
        )
        if variant_strategy == "all_angles":
            complete_groups = _accepted_complete_groups(
                round_drafts,
                round_verdicts,
                seen_prompts,
            )
            round_accepted = _trim_groups_to_shortfall(complete_groups, missing)
        else:
            round_accepted = [
                draft
                for draft in accepted_drafts(round_drafts, round_verdicts)
                if draft.prompt_text not in seen_prompts
            ]
        if trim_to_total and len(round_accepted) > missing:
            round_accepted = round_accepted[:missing]
        seen_prompts.update(draft.prompt_text for draft in round_accepted)

        all_drafts.extend(round_drafts)
        all_verdicts.extend(round_verdicts)
        accepted.extend(round_accepted)

        round_summary = validation_summary(round_verdicts)
        logger.info(
            "interview_generation_round",
            round=attempt,
            requested=request_count,
            produced=len(round_drafts),
            validation_accepted=round_summary["accepted"],
            validation_rejected=round_summary["rejected"],
            failure_codes=round_summary["failures"],
            new_accepted=len(round_accepted),
            accepted_total=len(accepted),
            target=target_count,
        )

        if on_progress is not None:
            await on_progress(min(len(accepted), target_count), target_count)

        if not round_drafts or (not round_accepted and variant_strategy != "all_angles"):
            break

    logger.info(
        "interview_generation_backfill_complete",
        accepted=len(accepted),
        target=target_count,
        backfill_rounds=backfill_rounds,
        drafts_total=len(all_drafts),
        rejected_total=len(all_drafts) - len(accepted),
    )

    return all_drafts, all_verdicts, accepted, backfill_rounds


__all__ = [
    "MAX_BACKFILL_ATTEMPTS",
    "accepted_drafts",
    "generate_with_backfill",
    "validation_summary",
]
