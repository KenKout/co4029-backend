"""Quiz retrieval stage (T5.4) — extracted from god file lines 961-1163.

Composes the T2.9 retrieval primitives (``vector_search``,
``mmr_diversify``, ``retrieve_kg_context_for_anchors``) plus the
quiz-specific anchor builder. The public entry point is
:func:`retrieve_chunks`; :func:`build_query_anchors` is re-exported for
callers that want to inspect the anchor list separately (e.g. audit /
debug surfaces).
"""

from __future__ import annotations

from abridgeai.features.quizzes.ai.stages.retrieval.anchors import build_query_anchors
from abridgeai.features.quizzes.ai.stages.retrieval.logic import retrieve_chunks
from abridgeai.features.quizzes.ai.stages.retrieval.metadata import retrieval_metadata

__all__ = [
    "build_query_anchors",
    "retrieval_metadata",
    "retrieve_chunks",
]
