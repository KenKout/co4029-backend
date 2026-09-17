"""Hybrid retrieval for one anchor: vector + BM25, fused.

The function is short and does one thing that matters. Its two legs
return different kinds of score: the semantic leg gives a real cosine
distance, and the BM25 leg gives none at all. Everything downstream --
MMR diversification, the cross-encoder rerank -- orders by ``distance``,
so a chunk that only BM25 found needs one synthesized, and it has to point
the same way round.

``distance`` is smaller-is-better; ``fused_score`` is larger-is-better. The
conversion is ``1 - fused_score``, and inverting it would send exactly the
chunks BM25 contributed to the back of the queue -- which is the half of
the hybrid retrieval that keyword matches were added for, silently
disabled with every other part still reporting success.

Both searches and the fusion are stubbed; no database.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from abridgeai.features.quizzes.ai.stages.retrieval import hybrid as hybrid_module
from abridgeai.features.quizzes.ai.stages.retrieval.hybrid import hybrid_search_for_anchor


def _fused(
    *,
    distance: float | None,
    fused_score: float,
    content: str = "passage",
    embedding: list[float] | None = None,
    chunk_id: UUID | None = None,
) -> SimpleNamespace:
    """A ``FusedChunk``-shaped row as the fusion primitive returns it."""
    return SimpleNamespace(
        chunk_id=chunk_id or uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=content,
        fused_score=fused_score,
        sources=frozenset({"bm25"} if distance is None else {"vector"}),
        distance=distance,
        embedding=embedding,
        metadata={"content_role": "body"},
    )


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what each leg was asked for, and control what fusion returns."""
    calls: dict[str, Any] = {"fused_rows": []}

    async def _vector(db, embedding, **kwargs):
        calls["vector"] = {"embedding": embedding, **kwargs}
        return ["semantic-hits"]

    async def _bm25(db, text, **kwargs):
        calls["bm25"] = {"text": text, **kwargs}
        return ["bm25-hits"]

    def _fusion(semantic, bm25, **kwargs):
        calls["fusion"] = {"semantic": semantic, "bm25": bm25, **kwargs}
        return calls["fused_rows"]

    monkeypatch.setattr(hybrid_module, "vector_search", _vector)
    monkeypatch.setattr(hybrid_module, "bm25_search", _bm25)
    monkeypatch.setattr(hybrid_module, "reciprocal_rank_fusion", _fusion)
    return calls


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        hybrid_recall_k=150,
        hybrid_semantic_weight=0.8,
        hybrid_bm25_weight=0.2,
    )


async def _search(stubs: dict[str, Any], **over: Any):
    kwargs: dict[str, Any] = {
        "anchor_text": "what is photosynthesis",
        "embedding": [0.1, 0.2, 0.3],
        "course_id": uuid4(),
        "lesson_ids": None,
        "settings": _settings(),
    }
    kwargs.update(over)
    return await hybrid_search_for_anchor(SimpleNamespace(), **kwargs)


class TestBothLegsAreAsked:
    async def test_the_vector_leg_reuses_the_precomputed_embedding(
        self, stubs: dict[str, Any]
    ) -> None:
        """The orchestrator already paid for this embedding; re-embedding
        the anchor here would double the cost of every hybrid query."""
        embedding = [0.4, 0.5, 0.6]

        await _search(stubs, embedding=embedding)

        assert stubs["vector"]["embedding"] == embedding

    async def test_the_vector_leg_asks_for_embeddings_back(
        self, stubs: dict[str, Any]
    ) -> None:
        """MMR runs downstream and needs the vectors.

        The column is skipped by default to save bandwidth, so forgetting
        the opt-in here does not fail -- MMR just falls back to relevance
        ordering and the diversification quietly stops happening.
        """
        await _search(stubs)

        assert stubs["vector"]["include_embeddings"] is True

    async def test_the_keyword_leg_gets_the_anchor_text(
        self, stubs: dict[str, Any]
    ) -> None:
        """BM25 matches words, so it needs the query, not the vector."""
        await _search(stubs, anchor_text="chloroplast structure")

        assert stubs["bm25"]["text"] == "chloroplast structure"

    async def test_both_legs_recall_the_same_depth(self, stubs: dict[str, Any]) -> None:
        """Reciprocal-rank fusion compares positions between the two lists.

        Different depths would make the shorter leg's tail look absent
        rather than lower-ranked, and bias the fusion toward whichever list
        was allowed to be longer.
        """
        await _search(stubs)

        assert stubs["vector"]["top_k"] == 150
        assert stubs["bm25"]["top_k"] == 150

    async def test_both_legs_are_scoped_to_the_same_material(
        self, stubs: dict[str, Any]
    ) -> None:
        """A leg that ignored the scope would pull chunks from another
        course into a quiz built for this one."""
        course_id, lesson_ids = uuid4(), [uuid4(), uuid4()]

        await _search(stubs, course_id=course_id, lesson_ids=lesson_ids)

        for leg in ("vector", "bm25"):
            assert stubs[leg]["course_id"] == course_id
            assert stubs[leg]["lesson_ids"] == lesson_ids

    async def test_the_configured_weights_reach_the_fusion(
        self, stubs: dict[str, Any]
    ) -> None:
        """They are the tuning knob for how much keyword matching counts;
        hardcoding them here would make the settings inert."""
        await _search(stubs)

        assert stubs["fusion"]["semantic_weight"] == 0.8
        assert stubs["fusion"]["bm25_weight"] == 0.2
        assert stubs["fusion"]["semantic"] == ["semantic-hits"]
        assert stubs["fusion"]["bm25"] == ["bm25-hits"]


class TestTheDistanceHandedDownstream:
    async def test_a_semantic_hit_keeps_its_real_cosine_distance(
        self, stubs: dict[str, Any]
    ) -> None:
        """It is a true measure and downstream re-scoring benefits from
        the real value rather than a rank-derived proxy."""
        stubs["fused_rows"] = [_fused(distance=0.17, fused_score=0.9)]

        [chunk] = await _search(stubs)

        assert chunk.distance == 0.17

    async def test_a_keyword_only_hit_gets_a_synthesized_distance(
        self, stubs: dict[str, Any]
    ) -> None:
        """BM25 never computed a vector distance, and ``None`` would break
        every downstream sort."""
        stubs["fused_rows"] = [_fused(distance=None, fused_score=0.75)]

        [chunk] = await _search(stubs)

        assert chunk.distance == pytest.approx(0.25)

    async def test_the_synthesis_points_the_same_way_as_a_real_distance(
        self, stubs: dict[str, Any]
    ) -> None:
        """The direction is the whole point.

        ``distance`` sorts ascending and ``fused_score`` descending, so the
        better-scoring keyword hit has to come out with the *smaller*
        distance. Inverted, every BM25-only chunk sinks to the bottom and
        the keyword leg stops contributing while still appearing to run.
        """
        better = _fused(distance=None, fused_score=0.9, content="better")
        worse = _fused(distance=None, fused_score=0.2, content="worse")
        stubs["fused_rows"] = [better, worse]

        chunks = await _search(stubs)

        assert chunks[0].content == "better"
        assert chunks[0].distance < chunks[1].distance

    async def test_a_zero_distance_is_kept_rather_than_synthesized(
        self, stubs: dict[str, Any]
    ) -> None:
        """A perfect semantic match has distance ``0.0``, which is falsey.

        The check is ``is not None`` precisely so the best possible hit is
        not mistaken for a missing measurement and rewritten from its
        fused score.
        """
        stubs["fused_rows"] = [_fused(distance=0.0, fused_score=0.5)]

        [chunk] = await _search(stubs)

        assert chunk.distance == 0.0


class TestWhatReachesTheCaller:
    async def test_the_fusion_ordering_is_preserved(
        self, stubs: dict[str, Any]
    ) -> None:
        """Fusion already ranked these; re-sorting here would discard the
        weighting it was asked to apply."""
        rows = [_fused(distance=None, fused_score=s, content=f"c{i}")
                for i, s in enumerate((0.9, 0.5, 0.1))]
        stubs["fused_rows"] = rows

        chunks = await _search(stubs)

        assert [c.content for c in chunks] == ["c0", "c1", "c2"]

    async def test_the_embedding_travels_with_a_semantic_hit(
        self, stubs: dict[str, Any]
    ) -> None:
        """MMR needs it. Dropping it here is the same silent downgrade as
        forgetting the opt-in on the vector leg."""
        stubs["fused_rows"] = [_fused(distance=0.2, fused_score=0.8, embedding=[0.1, 0.2])]

        [chunk] = await _search(stubs)

        assert chunk.embedding == [0.1, 0.2]

    async def test_a_keyword_only_hit_carries_no_embedding(
        self, stubs: dict[str, Any]
    ) -> None:
        """BM25 does not return vectors, and the MMR primitive tolerates
        the gap by falling back to relevance ordering for that chunk."""
        stubs["fused_rows"] = [_fused(distance=None, fused_score=0.6)]

        [chunk] = await _search(stubs)

        assert chunk.embedding is None

    async def test_the_metadata_blob_is_passed_through_whole(
        self, stubs: dict[str, Any]
    ) -> None:
        """The role-aware filter downstream reads
        ``metadata['semantic']['content_role']`` with a fallback to the
        top-level key, so precedence is resolved there -- which means the
        whole blob has to arrive intact.
        """
        row = _fused(distance=0.3, fused_score=0.7)
        row.metadata = {"content_role": "summary", "semantic": {"content_role": "body"}}
        stubs["fused_rows"] = [row]

        [chunk] = await _search(stubs)

        assert chunk.metadata == {
            "content_role": "summary",
            "semantic": {"content_role": "body"},
        }

    async def test_the_identifying_columns_survive_the_reshape(
        self, stubs: dict[str, Any]
    ) -> None:
        """The chunk id is what a generated question cites as its source."""
        row = _fused(distance=0.3, fused_score=0.7, content="the passage")
        stubs["fused_rows"] = [row]

        [chunk] = await _search(stubs)

        assert chunk.chunk_id == row.chunk_id
        assert chunk.material_version_id == row.material_version_id
        assert chunk.course_id == row.course_id
        assert chunk.lesson_id == row.lesson_id
        assert chunk.content == "the passage"

    async def test_an_empty_fusion_yields_an_empty_list(
        self, stubs: dict[str, Any]
    ) -> None:
        """Neither leg found anything: the caller gets nothing rather than
        a row of placeholders."""
        stubs["fused_rows"] = []

        assert await _search(stubs) == []
