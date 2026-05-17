"""Interview follow-up stage (T6.7) — runtime, not pipeline-time.

Public surface:

* :func:`maybe_generate_followup` — orchestrator entry point used by
  ``take_session_step`` in T6.13. Returns the follow-up question text or
  ``None`` when no follow-up should be asked.
* :func:`parse_followup_response` / :class:`FollowupVerdict` — pure
  parser exported for unit tests.
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.followup.logic import (
    FOLLOWUP_STAGE_NAME,
    maybe_generate_followup,
)
from abridgeai.features.interviews.ai.stages.followup.parsers import (
    FollowupVerdict,
    parse_followup_response,
)

__all__ = [
    "FOLLOWUP_STAGE_NAME",
    "FollowupVerdict",
    "maybe_generate_followup",
    "parse_followup_response",
]
