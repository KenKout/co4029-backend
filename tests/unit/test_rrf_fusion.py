"""Unit tests for the weighted Reciprocal Rank Fusion helper (Phase 3).

Mirrors the Anthropic cookbook's ``retrieve_advanced`` semantics:
weighted ``1/(rank+1)`` combination of two ranked lists, with the
union of chunk ids surfaced and ranked by descending fused score.
"""

from __future__ import annotations

from uuid import uuid4

from abridgeai.ai.retrieval import ChunkWithDistance, ChunkWithRank
from abridgeai.ai.retrieval.fusion import FusedChunk, reciprocal_rank_fusion


def _semantic(chunk_id, distance: float) -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=chunk_id,
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=f"semantic-{chunk_id}",
        distance=distance,
        embedding=[1.0, 0.0],
    )


def _bm25(chunk_id, rank: float) -> ChunkWithRank:
    return ChunkWithRank(
        chunk_id=chunk_id,
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=f"bm25-{chunk_id}",
        rank=rank,
    )


def test_rrf_returns_empty_when_both_lists_empty() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_promotes_dual_source_chunks_above_singleton_hits() -> None:
    """A chunk in BOTH lists should outrank a chunk in only one."""
    common_id = uuid4()
    only_semantic_id = uuid4()
    only_bm25_id = uuid4()

    semantic = [
        _semantic(common_id, distance=0.1),  # rank 0 in semantic
        _semantic(only_semantic_id, distance=0.2),  # rank 1
    ]
    bm25 = [
        _semantic(common_id, distance=0.0),  # placeholder
        # actual bm25 entry — common_id at rank 0
    ]
    bm25 = [
        _bm25(common_id, rank=10.0),  # rank 0 in bm25
        _bm25(only_bm25_id, rank=5.0),  # rank 1
    ]

    fused = reciprocal_rank_fusion(
        semantic,
        bm25,
        semantic_weight=0.8,
        bm25_weight=0.2,
    )

    assert [f.chunk_id for f in fused][0] == common_id
    common_chunk = next(f for f in fused if f.chunk_id == common_id)
    assert common_chunk.sources == frozenset({"vector", "bm25"})


def test_rrf_weights_match_anthropic_cookbook_formula() -> None:
    """semantic at rank0 with weight 0.8 → score 0.8 * (1/1) = 0.8.
    bm25 at rank0 with weight 0.2 → score 0.2 * (1/1) = 0.2.
    Combined chunk score = 1.0 (within float tolerance)."""
    chunk_id = uuid4()
    fused = reciprocal_rank_fusion(
        [_semantic(chunk_id, distance=0.1)],
        [_bm25(chunk_id, rank=9.5)],
        semantic_weight=0.8,
        bm25_weight=0.2,
    )
    assert len(fused) == 1
    assert abs(fused[0].fused_score - 1.0) < 1e-9
    assert fused[0].sources == frozenset({"vector", "bm25"})


def test_rrf_adjusts_to_extreme_weights() -> None:
    """semantic_weight=0 should make BM25 ordering dominate entirely."""
    sem_only = uuid4()
    bm25_only = uuid4()

    fused = reciprocal_rank_fusion(
        [_semantic(sem_only, distance=0.1)],
        [_bm25(bm25_only, rank=9.0)],
        semantic_weight=0.0,
        bm25_weight=1.0,
    )
    # bm25_only should be first (only it has non-zero score)
    assert fused[0].chunk_id == bm25_only
    sem_chunk = next(f for f in fused if f.chunk_id == sem_only)
    assert sem_chunk.fused_score == 0.0


def test_rrf_preserves_embedding_and_distance_from_semantic_leg() -> None:
    """Downstream MMR needs the embedding from vector_search rows."""
    common = uuid4()
    fused = reciprocal_rank_fusion(
        [_semantic(common, distance=0.1)],
        [_bm25(common, rank=5.0)],
    )
    assert fused[0].embedding == [1.0, 0.0]
    assert fused[0].distance == 0.1


def test_rrf_bm25_only_chunk_has_no_embedding() -> None:
    """BM25-only hits surface a None embedding (MMR tolerates this)."""
    bm25_only = uuid4()
    fused = reciprocal_rank_fusion(
        [],
        [_bm25(bm25_only, rank=5.0)],
    )
    assert fused[0].embedding is None
    assert fused[0].distance is None
    assert fused[0].sources == frozenset({"bm25"})


def test_rrf_returns_frozen_dataclass() -> None:
    chunk_id = uuid4()
    fused = reciprocal_rank_fusion([_semantic(chunk_id, distance=0.1)], [])
    assert isinstance(fused[0], FusedChunk)
    try:
        fused[0].fused_score = 9.9  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("FusedChunk should be frozen")


def test_rrf_orders_purely_by_fused_score_descending() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    fused = reciprocal_rank_fusion(
        [_semantic(a, distance=0.1), _semantic(b, distance=0.2), _semantic(c, distance=0.3)],
        [_bm25(c, rank=9.0), _bm25(b, rank=5.0)],
        semantic_weight=0.5,
        bm25_weight=0.5,
    )
    scores = [f.fused_score for f in fused]
    assert scores == sorted(scores, reverse=True)
