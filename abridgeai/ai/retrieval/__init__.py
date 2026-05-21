"""Retrieval primitives for AI generation pipelines (T2.9).

Sub-modules:

  * :mod:`~abridgeai.ai.retrieval.pgvector` — cosine-distance vector
    search over ``document_chunks``.
  * :mod:`~abridgeai.ai.retrieval.bm25` — Postgres tsvector lexical
    search (Phase 3 contextual-RAG hybrid retrieval).
  * :mod:`~abridgeai.ai.retrieval.mmr` — Maximal Marginal Relevance
    diversification.
  * :mod:`~abridgeai.ai.retrieval.kg` — re-export of the knowledge-graph
    anchor traversal from :mod:`abridgeai.ai.knowledge_graph`.

Quiz / interview generation stages (Phase 5/6) import from this package
so they do not have to know which sub-module owns each primitive.
"""

from __future__ import annotations

from abridgeai.ai.retrieval.bm25 import ChunkWithRank, bm25_search
from abridgeai.ai.retrieval.kg import retrieve_kg_context_for_anchors
from abridgeai.ai.retrieval.mmr import mmr_diversify
from abridgeai.ai.retrieval.pgvector import ChunkWithDistance, vector_search

__all__ = [
    "ChunkWithDistance",
    "ChunkWithRank",
    "bm25_search",
    "mmr_diversify",
    "retrieve_kg_context_for_anchors",
    "vector_search",
]
