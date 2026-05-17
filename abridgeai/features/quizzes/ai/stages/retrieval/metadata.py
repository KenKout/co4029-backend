"""Retrieval audit metadata helpers (T5.4).

Ports ``_retrieval_metadata`` from
``backend/app/ai/haystack/pipelines/quiz_generation.py:1157-1162``.

Used by the orchestrator to attach a small JSON-serializable summary of
the retrieval result to the pipeline-run audit row, so QA / regression
can replay how a question was grounded.
"""

from __future__ import annotations

from typing import Any

from abridgeai.ai.retrieval import ChunkWithDistance


def retrieval_metadata(
    chunks: list[ChunkWithDistance],
    *,
    anchors: list[str],
    primary_embedding: list[float] | None = None,
    kg_context_enabled: bool = True,
) -> dict[str, Any]:
    """Return a JSON-serializable audit summary for a retrieval result.

    Parameters
    ----------
    chunks
        Final selected chunks (post-MMR).
    anchors
        The anchor list that drove the retrieval (mirrors the FR-11
        replay requirement).
    primary_embedding
        First-anchor embedding; only its presence is recorded (length +
        non-zero check) — vectors themselves are too large for audit.
    kg_context_enabled
        Whether the orchestrator was configured to consult the KG.
    """

    primary = primary_embedding or []
    return {
        "chunk_count": len(chunks),
        "anchor_count": len(anchors),
        "anchors": list(anchors),
        "strategy": "vector_mmr" if any(primary) else "fallback",
        "kg_context_enabled": bool(kg_context_enabled),
        "embedding_dimensions": len(primary),
        "source_chunk_ids": [str(chunk.chunk_id) for chunk in chunks],
    }


__all__ = ["retrieval_metadata"]
