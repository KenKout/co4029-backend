"""Interview retrieval stage (T6.4) — chunks + KG context aggregator.

Composes :mod:`abridgeai.ai.retrieval` primitives + the interview-flavour
anchor builder. Public entry point is :func:`retrieve_interview_context`;
:func:`build_interview_anchors` is re-exported for callers that want to
inspect the intermediate signals (audit / debug surfaces).
"""

from __future__ import annotations

from abridgeai.features.interviews.ai.stages.retrieval.anchors import (
    build_interview_anchors,
)
from abridgeai.features.interviews.ai.stages.retrieval.logic import (
    InterviewRetrievalContext,
    retrieve_interview_context,
)
from abridgeai.features.interviews.ai.stages.retrieval.metadata import (
    retrieval_metadata,
)

__all__ = [
    "InterviewRetrievalContext",
    "build_interview_anchors",
    "retrieval_metadata",
    "retrieve_interview_context",
]
