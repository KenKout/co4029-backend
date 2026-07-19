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

from abridgeai.features.interviews.ai.stages.generation import (
    generate_interview_questions,
)
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


async def generate_with_backfill(
    db: AsyncSession,
    *,
    state: Any,
    config: InterviewConfig,
    context: InterviewRetrievalContext,
    outcomes: list[Any],
    target_count: int,
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

        round_drafts = await generate_interview_questions(
            db,
            run=state,
            config=config,
            context=cast("Any", context),
            outcomes=cast("Any", outcomes),
            override_question_count=request_count,
            avoid_prompts=[d.prompt_text for d in accepted],
        )
        round_verdicts = await validate_interview_questions(
            db,
            run=state,
            config=config,
            drafts=cast("Any", round_drafts),
            context=cast("Any", context),
        )
        round_accepted = [
            d
            for d in accepted_drafts(round_drafts, round_verdicts)
            if d.prompt_text not in seen_prompts
        ]
        seen_prompts.update(d.prompt_text for d in round_accepted)

        all_drafts.extend(round_drafts)
        all_verdicts.extend(round_verdicts)
        accepted.extend(round_accepted)

        if on_progress is not None:
            await on_progress(min(len(accepted), target_count), target_count)

        if not round_drafts or not round_accepted:
            break

    return all_drafts, all_verdicts, accepted, backfill_rounds


__all__ = [
    "MAX_BACKFILL_ATTEMPTS",
    "accepted_drafts",
    "generate_with_backfill",
    "validation_summary",
]
