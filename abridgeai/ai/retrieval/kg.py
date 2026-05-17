"""Single-import surface for KG retrieval primitive.

This module re-exports :func:`retrieve_kg_context_for_anchors` from
:mod:`abridgeai.ai.knowledge_graph.retrieval` so that retrieval
consumers (quiz / interview generation stages) only need to import from
``abridgeai.ai.retrieval``.
"""

from __future__ import annotations

from abridgeai.ai.knowledge_graph.retrieval import retrieve_kg_context_for_anchors

__all__ = ["retrieve_kg_context_for_anchors"]
