"""Retrieval audit metadata helpers (T6.4).

Mirrors :mod:`abridgeai.features.quizzes.ai.stages.retrieval.metadata`
but adds the interview-specific ``weak_topic_chunk_count`` and
``kg_concept_count`` fields that audit the student-weakness lookup +
KG fan-out used by the generation stage.
"""

from __future__ import annotations

from typing import Any

from abridgeai.ai.knowledge_graph.schemas import Concept
from abridgeai.ai.retrieval import ChunkWithDistance


def retrieval_metadata(
    chunks: list[ChunkWithDistance],
    *,
    anchors: list[str],
    kg_concepts: list[Concept],
    weak_topic_chunks: list[ChunkWithDistance],
    primary_embedding: list[float] | None = None,
    kg_context_enabled: bool = True,
    student_id: str | None = None,
) -> dict[str, Any]:
    """JSON-serializable summary of an interview retrieval result."""

    primary = primary_embedding or []
    return {
        "chunk_count": len(chunks),
        "anchor_count": len(anchors),
        "anchors": list(anchors),
        "strategy": "vector_mmr" if any(primary) else "fallback",
        "kg_context_enabled": bool(kg_context_enabled),
        "kg_concept_count": len(kg_concepts),
        "embedding_dimensions": len(primary),
        "source_chunk_ids": [str(chunk.chunk_id) for chunk in chunks],
        "weak_topic_chunk_count": len(weak_topic_chunks),
        "weak_topic_chunk_ids": [str(c.chunk_id) for c in weak_topic_chunks],
        "student_id": student_id,
    }


__all__ = ["retrieval_metadata"]
