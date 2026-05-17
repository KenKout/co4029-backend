"""Gap Report stage (T6.9) — aBridgeAI's signature theory-vs-practice synthesis.

Public surface:

* :func:`generate_gap_report` — orchestrator entry point used by T6.11
  services after a session is evaluated.
* :class:`GapReportDraft` / :class:`StudyPlanItem` — typed return shape
  consumed by T6.11 to persist a ``gap_reports`` row.
* :func:`parse_gap_report_response` — pure parser exported for tests.
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.gap_report.logic import (
    GAP_REPORT_STAGE_NAME,
    GapReportDraft,
    StudyPlanItem,
    generate_gap_report,
)
from abridgeai.features.interviews.ai.stages.gap_report.parsers import (
    parse_gap_report_response,
)

__all__ = [
    "GAP_REPORT_STAGE_NAME",
    "GapReportDraft",
    "StudyPlanItem",
    "generate_gap_report",
    "parse_gap_report_response",
]
