from __future__ import annotations

from uuid import uuid4

from abridgeai.ai.retrieval import ChunkWithDistance, mmr_diversify


def _chunk(distance: float, embedding: list[float], content: str = "x") -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=content,
        distance=distance,
        embedding=embedding,
    )


def test_empty_input_returns_empty_list():
    assert mmr_diversify([], top_k=5) == []


def test_top_k_zero_returns_empty():
    chunks = [_chunk(0.1, [1.0, 0.0, 0.0, 0.0])]
    assert mmr_diversify(chunks, top_k=0) == []


def test_top_k_ge_len_returns_all_in_order():
    a = _chunk(0.10, [1.0, 0.0, 0.0, 0.0], content="A")
    b = _chunk(0.20, [0.0, 1.0, 0.0, 0.0], content="B")
    out = mmr_diversify([a, b], top_k=5)
    assert [c.content for c in out] == ["A", "B"]


def test_mmr_diversify_balances_relevance_and_diversity():
    near1 = _chunk(0.05, [1.0, 0.0, 0.0, 0.0], content="near1")
    near2 = _chunk(0.06, [0.99, 0.01, 0.0, 0.0], content="near2")
    near3 = _chunk(0.07, [0.98, 0.02, 0.0, 0.0], content="near3")
    div1 = _chunk(0.40, [0.0, 0.0, 1.0, 0.0], content="div1")
    div2 = _chunk(0.50, [0.0, 0.0, 0.0, 1.0], content="div2")

    out = mmr_diversify([near1, near2, near3, div1, div2], top_k=3, lambda_diversity=0.5)
    contents = [c.content for c in out]

    assert len(out) == 3
    assert contents[0] == "near1"
    assert "div1" in contents or "div2" in contents
    near_count = sum(1 for c in contents if c.startswith("near"))
    assert near_count <= 2


def test_mmr_lambda_one_returns_pure_relevance_order():
    near1 = _chunk(0.05, [1.0, 0.0, 0.0, 0.0], content="near1")
    near2 = _chunk(0.06, [0.99, 0.01, 0.0, 0.0], content="near2")
    near3 = _chunk(0.07, [0.98, 0.02, 0.0, 0.0], content="near3")
    div = _chunk(0.40, [0.0, 0.0, 1.0, 0.0], content="div")

    out = mmr_diversify([near1, near2, near3, div], top_k=3, lambda_diversity=1.0)
    assert [c.content for c in out] == ["near1", "near2", "near3"]


def test_mmr_lambda_zero_pure_diversity_picks_dissimilar():
    near1 = _chunk(0.05, [1.0, 0.0, 0.0, 0.0], content="near1")
    near2 = _chunk(0.06, [0.99, 0.01, 0.0, 0.0], content="near2")
    div_a = _chunk(0.40, [0.0, 0.0, 1.0, 0.0], content="div_a")
    div_b = _chunk(0.50, [0.0, 0.0, 0.0, 1.0], content="div_b")

    out = mmr_diversify([near1, near2, div_a, div_b], top_k=3, lambda_diversity=0.0)
    contents = [c.content for c in out]

    assert contents[0] == "near1"
    assert "near2" not in contents
    assert "div_a" in contents
    assert "div_b" in contents


def test_mmr_handles_missing_embeddings_without_error():
    a = ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=None,
        lesson_id=None,
        content="A",
        distance=0.1,
        embedding=None,
    )
    b = ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=None,
        lesson_id=None,
        content="B",
        distance=0.2,
        embedding=None,
    )
    out = mmr_diversify([a, b], top_k=2, lambda_diversity=0.5)
    assert [c.content for c in out] == ["A", "B"]


def test_mmr_lambda_clamped_to_unit_interval():
    a = _chunk(0.05, [1.0, 0.0, 0.0, 0.0], content="A")
    b = _chunk(0.40, [0.0, 0.0, 1.0, 0.0], content="B")
    out_high = mmr_diversify([a, b], top_k=2, lambda_diversity=5.0)
    out_low = mmr_diversify([a, b], top_k=2, lambda_diversity=-3.0)
    assert {c.content for c in out_high} == {"A", "B"}
    assert {c.content for c in out_low} == {"A", "B"}


def test_kg_retrieval_re_exported():
    from abridgeai.ai.retrieval import retrieve_kg_context_for_anchors

    assert callable(retrieve_kg_context_for_anchors)
