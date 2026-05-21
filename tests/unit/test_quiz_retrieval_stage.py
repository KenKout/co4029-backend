"""Unit tests for the quiz retrieval stage (T5.4).

Covers acceptance items in plan §5577-5598:
* MMR diversification path (mocked vector_search returns 20 hits → 12 selected)
* KG context lookup invoked when ``kg_context_enabled=True``
* KG context skipped when ``kg_context_enabled=False``
* Anchor builder ports the legacy precedence (focus_topics > KG > titles > quiz.title)
* No file in ``stages/retrieval/`` exceeds 250 LOC

The Quiz ORM is heavy enough that we use a SimpleNamespace stand-in:
``retrieve_chunks`` only reads ``quiz.module_id`` + ``quiz.title``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from abridgeai.ai.knowledge_graph.schemas import Concept, KGContext
from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.features.quizzes.ai.stages.retrieval import (
    build_query_anchors,
    retrieval_metadata,
    retrieve_chunks,
)


def _chunk(distance: float, content: str = "x") -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=content,
        distance=distance,
        embedding=[1.0 - distance, distance, 0.0, 0.0],
    )


def _quiz_stub(*, title: str = "Sample Quiz", module_id: object | None = None):
    return SimpleNamespace(title=title, module_id=module_id or uuid4())


@pytest.fixture
def fake_embedding_client() -> AsyncMock:
    client = AsyncMock()
    client.embed_query = AsyncMock(return_value=[0.1] * 8)
    return client


@pytest.mark.asyncio
async def test_retrieve_chunks_returns_diversified_results(
    fake_embedding_client: AsyncMock,
) -> None:
    db = AsyncMock()
    quiz = _quiz_stub()
    config = {"focus_topics": ["alpha", "beta"]}

    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(20)]

    with patch(
        "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
        AsyncMock(return_value=hits),
    ) as mock_vs:
        chunks, primary, anchors = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            final_top_k=12,
        )

    assert anchors == ["alpha", "beta"]
    assert mock_vs.await_count == 2
    assert len(chunks) == 12
    assert len({c.chunk_id for c in chunks}) == 12
    assert primary == [0.1] * 8


@pytest.mark.asyncio
async def test_retrieve_chunks_includes_kg_context_when_enabled(
    fake_embedding_client: AsyncMock,
) -> None:
    db = AsyncMock()
    quiz = _quiz_stub()
    lesson_id = uuid4()
    config = {"source_lesson_ids": [str(lesson_id)]}

    kg = KGContext(concepts=[Concept(name="Recursion"), Concept(name="Stack")], enabled=True)

    with (
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=kg),
        ) as mock_kg,
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=[_chunk(0.1)]),
        ),
    ):
        chunks, _, anchors = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=True,
            embedding_client=fake_embedding_client,
        )

    mock_kg.assert_awaited_once()
    assert anchors == ["Recursion", "Stack"]
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_retrieve_chunks_skips_kg_when_disabled(
    fake_embedding_client: AsyncMock,
) -> None:
    db = AsyncMock()
    quiz = _quiz_stub(title="Quiz Title")
    lesson_id = uuid4()
    config = {"source_lesson_ids": [str(lesson_id)]}

    from unittest.mock import MagicMock

    titles_result = MagicMock()
    titles_result.all.return_value = [("Lesson 1",), ("Lesson 2",)]
    db.execute = AsyncMock(return_value=titles_result)

    with (
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(),
        ) as mock_kg,
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=[_chunk(0.1)]),
        ),
    ):
        _, _, anchors = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
        )

    mock_kg.assert_not_awaited()
    assert anchors == ["Lesson 1", "Lesson 2"]


@pytest.mark.asyncio
async def test_build_query_anchors_focus_topics_take_precedence() -> None:
    db = AsyncMock()
    quiz = _quiz_stub()
    config = {
        "focus_topics": ["Hashing", " B-Trees ", "", 42],
        "source_lesson_ids": [str(uuid4())],
    }

    with patch(
        "abridgeai.features.quizzes.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
        AsyncMock(side_effect=AssertionError("must not be called when focus_topics set")),
    ):
        anchors = await build_query_anchors(db, quiz, config)

    assert anchors == ["Hashing", "B-Trees"]


@pytest.mark.asyncio
async def test_build_query_anchors_question_hint_short_circuits() -> None:
    db = AsyncMock()
    quiz = _quiz_stub()
    anchors = await build_query_anchors(
        db,
        quiz,
        {"focus_topics": ["ignored"]},
        question_hint="What is a B-tree?",
    )
    assert anchors == ["What is a B-tree?"]


@pytest.mark.asyncio
async def test_retrieve_chunks_returns_empty_when_no_anchors() -> None:
    db = AsyncMock()
    quiz = SimpleNamespace(title="", module_id=None)

    from unittest.mock import MagicMock

    titles_result = MagicMock()
    titles_result.all.return_value = []
    db.execute = AsyncMock(return_value=titles_result)

    chunks, primary, anchors = await retrieve_chunks(
        db,
        run_id=uuid4(),
        quiz=quiz,
        config={},
        kg_context_enabled=False,
        embedding_client=AsyncMock(),
    )
    assert chunks == []
    assert primary == []
    assert anchors == []


def test_retrieval_metadata_records_chunk_ids_and_strategy() -> None:
    chunks = [_chunk(0.1), _chunk(0.2)]
    meta = retrieval_metadata(
        chunks,
        anchors=["a", "b"],
        primary_embedding=[0.5, 0.5],
        kg_context_enabled=True,
    )
    assert meta["chunk_count"] == 2
    assert meta["anchor_count"] == 2
    assert meta["anchors"] == ["a", "b"]
    assert meta["strategy"] == "vector_mmr"
    assert meta["embedding_dimensions"] == 2
    assert meta["kg_context_enabled"] is True
    assert len(meta["source_chunk_ids"]) == 2


def test_retrieval_metadata_falls_back_strategy_without_embedding() -> None:
    meta = retrieval_metadata([], anchors=[], primary_embedding=None)
    assert meta["strategy"] == "fallback"
    assert meta["chunk_count"] == 0
    assert meta["embedding_dimensions"] == 0


def test_no_god_file_in_retrieval_stage() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "quizzes" / "ai" / "stages" / "retrieval"
    assert target.is_dir(), f"retrieval stage dir not found at {target}"
    for path in target.glob("*.py"):
        with path.open() as fh:
            line_count = sum(1 for _ in fh)
        assert line_count <= 250, f"{path.name} has {line_count} LOC > 250"


# ---------------------------------------------------------------------------
# Phase 4 — Voyage rerank wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_chunks_reorders_pool_when_rerank_client_injected(
    fake_embedding_client: AsyncMock,
) -> None:
    """When a rerank client is passed, the post-MMR pool is reordered by
    Voyage scores and capped at ``final_top_k``."""
    from abridgeai.ai.llm.voyage_rerank import RerankResult

    db = AsyncMock()
    quiz = _quiz_stub()
    config = {"focus_topics": ["alpha"]}

    # 20 vector hits → MMR widens pool (rerank path) → rerank reorders
    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(20)]

    rerank_client = AsyncMock()
    # Return scores that promote index 5 then 0 then 9 (out of natural order)
    rerank_client.rerank = AsyncMock(
        return_value=(
            [
                RerankResult(index=5, relevance_score=0.99),
                RerankResult(index=0, relevance_score=0.92),
                RerankResult(index=9, relevance_score=0.87),
                RerankResult(index=2, relevance_score=0.55),
            ],
            42,
        )
    )

    with patch(
        "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
        AsyncMock(return_value=hits),
    ):
        chunks, _, _ = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            rerank_client=rerank_client,
            final_top_k=4,
        )

    assert rerank_client.rerank.await_count == 1
    rerank_call = rerank_client.rerank.await_args
    assert rerank_call.args[0] == "alpha"  # primary anchor
    # Voyage saw at least final_top_k * RERANK_POOL_MULTIPLIER docs
    assert len(rerank_call.args[1]) >= 4
    assert len(chunks) == 4


@pytest.mark.asyncio
async def test_retrieve_chunks_falls_back_when_rerank_raises(
    fake_embedding_client: AsyncMock,
) -> None:
    """A provider error must not propagate — caller gets MMR-only top-K."""
    from abridgeai.ai.llm.errors import ProviderError

    db = AsyncMock()
    quiz = _quiz_stub()
    config = {"focus_topics": ["alpha"]}
    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(15)]

    rerank_client = AsyncMock()
    rerank_client.rerank = AsyncMock(side_effect=ProviderError("voyage 503"))

    with patch(
        "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
        AsyncMock(return_value=hits),
    ):
        chunks, _, _ = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            rerank_client=rerank_client,
            final_top_k=5,
        )

    assert rerank_client.rerank.await_count == 1
    assert len(chunks) == 5  # fallback path still returns final_top_k


@pytest.mark.asyncio
async def test_retrieve_chunks_skips_rerank_when_no_key_no_client(
    fake_embedding_client: AsyncMock,
) -> None:
    """Default settings have no Voyage key — MMR-only path runs."""
    db = AsyncMock()
    quiz = _quiz_stub()
    config = {"focus_topics": ["alpha"]}
    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(10)]

    with patch(
        "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
        AsyncMock(return_value=hits),
    ):
        chunks, _, _ = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            final_top_k=6,
        )

    assert len(chunks) == 6


# ---------------------------------------------------------------------------
# Phase 3 — Contextual BM25 hybrid retrieval
# ---------------------------------------------------------------------------


def _settings_with_hybrid(enabled: bool) -> object:
    """Build a minimal Settings stand-in for the hybrid knob."""
    from abridgeai.core.config import get_settings

    settings = get_settings()
    object.__setattr__(settings, "hybrid_bm25_enabled", enabled)
    return settings


@pytest.mark.asyncio
async def test_retrieve_chunks_dispatches_to_hybrid_when_enabled(
    fake_embedding_client: AsyncMock,
) -> None:
    """When hybrid_bm25_enabled=True the orchestrator calls
    hybrid_search_for_anchor instead of plain vector_search."""
    db = AsyncMock()
    quiz = _quiz_stub()
    config = {"focus_topics": ["alpha"]}

    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(8)]
    settings = _settings_with_hybrid(True)

    with (
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.logic.hybrid_search_for_anchor",
            AsyncMock(return_value=hits),
        ) as mock_hybrid,
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
            AsyncMock(side_effect=AssertionError("vector_search must not be called")),
        ),
    ):
        chunks, _, _ = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            final_top_k=4,
            settings=settings,
        )

    assert mock_hybrid.await_count == 1
    call_kwargs = mock_hybrid.await_args.kwargs
    assert call_kwargs["anchor_text"] == "alpha"
    assert call_kwargs["settings"] is settings
    assert len(chunks) == 4


@pytest.mark.asyncio
async def test_retrieve_chunks_uses_vector_only_when_hybrid_disabled(
    fake_embedding_client: AsyncMock,
) -> None:
    """Default settings (hybrid_bm25_enabled=False) skip the hybrid path."""
    db = AsyncMock()
    quiz = _quiz_stub()
    config = {"focus_topics": ["alpha"]}
    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(8)]
    settings = _settings_with_hybrid(False)

    with (
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.logic.hybrid_search_for_anchor",
            AsyncMock(side_effect=AssertionError("hybrid must not be called")),
        ),
        patch(
            "abridgeai.features.quizzes.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=hits),
        ) as mock_vs,
    ):
        chunks, _, _ = await retrieve_chunks(
            db,
            run_id=uuid4(),
            quiz=quiz,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            final_top_k=4,
            settings=settings,
        )

    assert mock_vs.await_count == 1
    assert len(chunks) == 4
