"""Telemetry helpers shared by quiz pipelines.

When the validator rejects every question and the pipeline aborts, the
audit row in ``ai_model_calls`` is rolled back along with the failed
transaction — losing the validator's verdict reasons. Log them
explicitly before raising so post-mortem diagnosis has the defect codes
and evidence excerpts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abridgeai.features.quizzes.ai.stages.validation.parsers import Verdict

_logger = logging.getLogger(__name__)


def log_validator_aborted_run(
    *,
    candidates: list[dict[str, Any]],
    rejected: list[Any],
    drops: list[Any],
    verdicts: list[Verdict],
    log_prefix: str,
) -> None:
    """Log a defect-code-bearing summary just before the pipeline raises.

    ``log_prefix`` differentiates which pipeline failed
    (``quiz_pipeline_aborted`` vs ``coverage_pipeline_aborted``) so a
    grep over worker logs partitions cleanly.
    """
    verdict_summary = [
        {
            "position": v.position,
            "verdict": v.verdict,
            "reasons": v.reasons,
            "evidence": (v.evidence_excerpt or "")[:120],
        }
        for v in verdicts
    ]
    _logger.warning(
        "%s_no_questions_survived: "
        "generated=%s rejected=%s dropped_by_dedup=%s verdicts=%s",
        log_prefix,
        len(candidates),
        len(rejected),
        len(drops),
        verdict_summary,
    )


__all__ = ["log_validator_aborted_run"]
