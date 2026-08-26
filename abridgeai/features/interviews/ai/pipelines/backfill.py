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


def _trim_to_shortfall(
    drafts: list[InterviewQuestionDraft],
    missing: int,
) -> list[InterviewQuestionDraft]:
    """Trim an overshooting variant round down to ``missing`` rows.

    A logical-unit backfill request returns whole angle groups (one draft
    per interviewer angle). When the accepted set exceeds the shortfall we
    keep rows spread across as many distinct ``question_type`` buckets as
    possible — dropping surplus duplicates of a type before dropping a
    type entirely — so whichever angle was missing stays represented.
    """
    by_type: dict[str, list[InterviewQuestionDraft]] = {}
    for d in drafts:
        by_type.setdefault(d.question_type, []).append(d)
    kept: list[InterviewQuestionDraft] = []
    # Round-robin one row per type until the shortfall is met.
    while len(kept) < missing and any(by_type.values()):
        for qtype in list(by_type):
            if len(kept) < missing and by_type[qtype]:
                kept.append(by_type[qtype].pop(0))
    return kept


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
        # (ceil), so passing the raw shortfall under-requests — a shortfall of
        # 2 rows becomes ceil(2/4)=1 logical question whose variants may not
        # include the missing angle (observed: bank short one behavioral).
        # Ask in LOGICAL units so every angle is re-requested, then trim the
        # surplus rows below so the run still lands exactly on target.
        trim_to_total = False
        if variant_strategy == "all_angles":
            request_count = -(-request_count // len(VARIANT_ANGLES)) * len(VARIANT_ANGLES)
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
        round_accepted = [
            d
            for d in accepted_drafts(round_drafts, round_verdicts)
            if d.prompt_text not in seen_prompts
        ]
        # Variant mode: a logical-unit backfill request yields a full angle set,
        # which may overshoot ``target_count``. Keep only the missing rows —
        # preferring the types still short so the bank stays balanced per role.
        if trim_to_total and len(round_accepted) > missing:
            round_accepted = _trim_to_shortfall(round_accepted, missing)
        seen_prompts.update(d.prompt_text for d in round_accepted)

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

        if not round_drafts or not round_accepted:
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
