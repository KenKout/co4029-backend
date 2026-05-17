"""Unit tests for the interview retrieval stage (T6.4).

Covers acceptance items in plan §6284-6322:
* Multi-anchor retrieval composes ``vector_search`` + MMR for module lessons.
* Student-weakness chunks pulled when ``run.config_json["student_id"]`` is set.
* Weak-topic lookup omitted (returns ``[]`` not ``None``) when student_id absent.
* KG concepts loaded when the module has lessons with KG anchors.
* No file in ``stages/retrieval/`` exceeds 300 LOC.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from abridgeai.ai.knowledge_graph.schemas import Concept, KGContext
from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.features.interviews.ai.stages.retrieval import (
    InterviewRetrievalContext,
    retrieve_interview_context,
)


def _chunk(distance: float = 0.1, content: str = "x") -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=content,
        distance=distance,
        embedding=[1.0 - distance, distance, 0.0, 0.0],
    )


def _config_stub(*, title: str = "Algorithms Interview", module_id: UUID | None = None):
    return SimpleNamespace(
        title=title,
        module_id=module_id or uuid4(),
        course_id=uuid4(),
    )


def _run_stub(
    *,
    config_json: dict | None = None,
    course_id: UUID | None = None,
):
    return SimpleNamespace(
        id=uuid4(),
        config_json=config_json or {},
        course_id=course_id,
    )


@pytest.fixture
def fake_embedding_client() -> AsyncMock:
    client = AsyncMock()
    client.embed_query = AsyncMock(return_value=[0.1] * 8)
    return client


@pytest.mark.asyncio
async def test_retrieves_chunks_for_module_lessons(
    fake_embedding_client: AsyncMock,
) -> None:
    """Happy path: 3 lessons → KG concepts drive anchors → chunks aggregated."""
    db = AsyncMock()
    config = _config_stub()
    run = _run_stub(config_json={"focus_topics": ["alpha", "beta"]})

    hits = [_chunk(distance=0.05 + i * 0.01, content=f"hit-{i}") for i in range(20)]

    with (
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=hits),
        ) as mock_vs,
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=KGContext(enabled=False)),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_ids",
            AsyncMock(return_value=[uuid4(), uuid4(), uuid4()]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.fetch_weak_topic_chunks",
            AsyncMock(return_value=[]),
        ),
    ):
        ctx = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
            final_top_k=12,
        )

    assert isinstance(ctx, InterviewRetrievalContext)
    assert ctx.anchors == ["alpha", "beta"]
    assert mock_vs.await_count == 2
    assert len(ctx.chunks) == 12
    assert len({c.chunk_id for c in ctx.chunks}) == 12
    assert ctx.query_embedding == [0.1] * 8
    assert ctx.metadata["chunk_count"] == 12
    assert ctx.metadata["anchor_count"] == 2


@pytest.mark.asyncio
async def test_includes_weak_topic_chunks_when_student_id_present(
    fake_embedding_client: AsyncMock,
) -> None:
    """Student id in run.config_json → weak-topic chunks merged into context."""
    db = AsyncMock()
    student_id = uuid4()
    config = _config_stub()
    run = _run_stub(
        config_json={
            "focus_topics": ["recursion"],
            "student_id": str(student_id),
        }
    )

    weak_chunks = [_chunk(content="weak-1"), _chunk(content="weak-2")]

    with (
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=[_chunk(0.1)]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=KGContext(enabled=False)),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_ids",
            AsyncMock(return_value=[uuid4()]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.fetch_weak_topic_chunks",
            AsyncMock(return_value=weak_chunks),
        ) as mock_weak,
    ):
        ctx = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
        )

    mock_weak.assert_awaited_once()
    call_kwargs = mock_weak.call_args.kwargs
    assert call_kwargs["student_id"] == student_id
    assert call_kwargs["module_id"] == config.module_id

    assert ctx.weak_topic_chunks == weak_chunks
    assert ctx.metadata["weak_topic_chunk_count"] == 2
    assert ctx.metadata["student_id"] == str(student_id)
    assert len(ctx.metadata["weak_topic_chunk_ids"]) == 2


@pytest.mark.asyncio
async def test_omits_weak_topics_when_no_student_id(
    fake_embedding_client: AsyncMock,
) -> None:
    """student_id missing → weak_topic_chunks is empty list (not None)."""
    db = AsyncMock()
    config = _config_stub()
    run = _run_stub(config_json={"focus_topics": ["binary search"]})

    with (
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=[_chunk(0.1)]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=KGContext(enabled=False)),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_ids",
            AsyncMock(return_value=[uuid4()]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.fetch_weak_topic_chunks",
            AsyncMock(return_value=[]),
        ) as mock_weak,
    ):
        ctx = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            kg_context_enabled=False,
            embedding_client=fake_embedding_client,
        )

    mock_weak.assert_not_called()
    assert ctx.weak_topic_chunks == []
    assert ctx.weak_topic_chunks is not None
    assert ctx.metadata["weak_topic_chunk_count"] == 0
    assert ctx.metadata["student_id"] is None


@pytest.mark.asyncio
async def test_kg_concepts_loaded_when_module_has_lessons_with_kg_anchors(
    fake_embedding_client: AsyncMock,
) -> None:
    """KG enabled + lessons present → retrieve_kg_context called, concepts populate anchors."""
    db = AsyncMock()
    config = _config_stub()
    run = _run_stub(config_json={})

    kg = KGContext(
        concepts=[Concept(name="Recursion"), Concept(name="Stack Overflow")],
        enabled=True,
    )

    with (
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=[_chunk(0.1)]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=kg),
        ) as mock_kg,
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_ids",
            AsyncMock(return_value=[uuid4(), uuid4()]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.fetch_weak_topic_chunks",
            AsyncMock(return_value=[]),
        ),
    ):
        ctx = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            kg_context_enabled=True,
            embedding_client=fake_embedding_client,
        )

    mock_kg.assert_awaited_once()
    assert ctx.anchors == ["Recursion", "Stack Overflow"]
    assert len(ctx.kg_concepts) == 2
    assert ctx.metadata["kg_concept_count"] == 2
    assert ctx.metadata["kg_context_enabled"] is True


@pytest.mark.asyncio
async def test_falls_back_to_lesson_titles_when_no_kg_or_focus_topics(
    fake_embedding_client: AsyncMock,
) -> None:
    db = AsyncMock()
    config = _config_stub()
    run = _run_stub(config_json={})

    titles_result = MagicMock()
    titles_result.all.return_value = [("Lesson 1",), ("Lesson 2",)]
    db.execute = AsyncMock(return_value=titles_result)

    with (
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.vector_search",
            AsyncMock(return_value=[_chunk(0.1)]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=KGContext(concepts=[], enabled=True)),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_ids",
            AsyncMock(return_value=[uuid4()]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.logic.fetch_weak_topic_chunks",
            AsyncMock(return_value=[]),
        ),
    ):
        ctx = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            kg_context_enabled=True,
            embedding_client=fake_embedding_client,
        )

    assert ctx.anchors == ["Lesson 1", "Lesson 2"]


@pytest.mark.asyncio
async def test_returns_empty_context_when_no_anchors() -> None:
    db = AsyncMock()
    config = _config_stub(title="", module_id=None)
    run = _run_stub(config_json={})

    with (
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_ids",
            AsyncMock(return_value=[]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors._module_lesson_titles",
            AsyncMock(return_value=[]),
        ),
        patch(
            "abridgeai.features.interviews.ai.stages.retrieval.anchors.retrieve_kg_context_for_anchors",
            AsyncMock(return_value=KGContext(enabled=False)),
        ),
    ):
        ctx = await retrieve_interview_context(
            db,
            run=run,
            config=config,
            kg_context_enabled=False,
            embedding_client=AsyncMock(),
        )

    assert ctx.chunks == []
    assert ctx.anchors == []
    assert ctx.query_embedding == []
    assert ctx.weak_topic_chunks == []
    assert ctx.metadata["chunk_count"] == 0


@pytest.mark.asyncio
async def test_weak_topic_sql_returns_empty_on_db_error() -> None:
    """Best-effort: SQL failure does not raise — empty list and run continues."""
    from abridgeai.features.interviews.ai.stages.retrieval.anchors import (
        fetch_weak_topic_chunks,
    )

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db gone"))

    out = await fetch_weak_topic_chunks(
        db,
        student_id=uuid4(),
        module_id=uuid4(),
    )

    assert out == []


@pytest.mark.asyncio
async def test_weak_topic_sql_parses_chunk_rows() -> None:
    from abridgeai.features.interviews.ai.stages.retrieval.anchors import (
        fetch_weak_topic_chunks,
    )

    db = AsyncMock()
    chunk_id_a, chunk_id_b = uuid4(), uuid4()
    rows = [
        {
            "chunk_id": chunk_id_a,
            "material_version_id": uuid4(),
            "course_id": uuid4(),
            "lesson_id": uuid4(),
            "content": "weak topic A",
        },
        {
            "chunk_id": chunk_id_b,
            "material_version_id": uuid4(),
            "course_id": None,
            "lesson_id": None,
            "content": "weak topic B",
        },
    ]
    mappings = MagicMock()
    mappings.all.return_value = rows
    result = MagicMock()
    result.mappings.return_value = mappings
    db.execute = AsyncMock(return_value=result)

    out = await fetch_weak_topic_chunks(
        db,
        student_id=uuid4(),
        module_id=uuid4(),
        limit=5,
    )

    assert len(out) == 2
    assert {c.chunk_id for c in out} == {chunk_id_a, chunk_id_b}
    assert all(c.embedding is None for c in out)


@pytest.mark.asyncio
async def test_weak_topic_lookup_skipped_for_zero_limit() -> None:
    from abridgeai.features.interviews.ai.stages.retrieval.anchors import (
        fetch_weak_topic_chunks,
    )

    db = AsyncMock()
    db.execute = AsyncMock()
    out = await fetch_weak_topic_chunks(
        db,
        student_id=uuid4(),
        module_id=uuid4(),
        limit=0,
    )
    assert out == []
    db.execute.assert_not_called()


def test_no_file_exceeds_300_loc() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "retrieval"
    assert target.is_dir(), f"retrieval stage dir not found at {target}"
    for path in target.glob("*.py"):
        with path.open() as fh:
            line_count = sum(1 for _ in fh)
        assert line_count <= 300, f"{path.name} has {line_count} LOC > 300"
