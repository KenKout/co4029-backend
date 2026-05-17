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
    target = (
        here
        / "abridgeai"
        / "features"
        / "quizzes"
        / "ai"
        / "stages"
        / "retrieval"
    )
    assert target.is_dir(), f"retrieval stage dir not found at {target}"
    for path in target.glob("*.py"):
        line_count = sum(1 for _ in path.open())
        assert line_count <= 250, f"{path.name} has {line_count} LOC > 250"
