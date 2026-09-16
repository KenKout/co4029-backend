from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.access_control import validate_catalog
from abridgeai.ai.chunking._glue import (
    _cosine,
    _to_list,
    absorb_tiny_windows,
    finalize_window,
    glue_by_similarity,
)
from abridgeai.ai.chunking.base import RawChunk
from abridgeai.ai.knowledge_graph.pruning import prune_superseded_chunk_graph
from abridgeai.ai.retrieval.bm25 import bm25_search
from abridgeai.features.career_paths.queries import readiness


def _chunk(
    index: int,
    text: str,
    tokens: int,
    *,
    role: str = "body",
    page: int | None = None,
    **metadata: object,
) -> RawChunk:
    return RawChunk(
        content=text,
        chunk_index=index,
        metadata={
            "token_count": tokens,
            "content_role": role,
            "page": page,
            **metadata,
        },
    )


class _Embedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts
        return self.vectors


class _SessionContext:
    def __init__(self, session: object) -> None:
        self.session = session

    async def __aenter__(self) -> object:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


def test_validate_catalog_cli_reports_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        validate_catalog,
        "load_catalog",
        lambda: SimpleNamespace(permissions=[1, 2]),
    )
    monkeypatch.setattr(
        validate_catalog,
        "load_role_seeds",
        lambda _catalog: SimpleNamespace(roles=[1]),
    )

    assert validate_catalog.main() == 0
    assert "2 permissions, 1 roles" in capsys.readouterr().out


def test_validate_catalog_cli_reports_loader_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fail() -> None:
        raise ValueError("duplicate permission")

    monkeypatch.setattr(validate_catalog, "load_catalog", _fail)

    assert validate_catalog.main() == 1
    assert "catalog validation FAILED: duplicate permission" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_glue_similarity_merges_only_compatible_neighbours() -> None:
    chunks = [
        _chunk(0, "alpha", 10, page=1, noise_flags=["header"]),
        _chunk(1, "beta", 12, page=2, noise_flags=["footer"]),
        _chunk(2, "summary", 5, role="summary", page=3),
    ]

    windows = await glue_by_similarity(
        chunks,
        _Embedder([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        min_window_tokens=0,
    )

    assert [window.content for window in windows] == ["alpha\n\nbeta", "summary"]
    assert windows[0].metadata["member_indices"] == [0, 1]
    assert windows[0].metadata["page_range"] == (1, 2)
    assert windows[0].metadata["noise_flags"] == ["header", "footer"]


@pytest.mark.asyncio
async def test_glue_fallback_absorbs_tiny_window_and_reindexes() -> None:
    windows = await glue_by_similarity(
        [_chunk(0, "tiny", 2), _chunk(1, "large", 40)],
        None,
        min_window_tokens=10,
        max_window_tokens=100,
    )

    assert len(windows) == 1
    assert windows[0].content == "tiny\n\nlarge"
    assert windows[0].chunk_index == 0
    assert windows[0].metadata["glue_group_id"] == 0


@pytest.mark.asyncio
async def test_glue_rejects_wrong_vector_count_and_handles_empty() -> None:
    assert await glue_by_similarity([], None) == []
    with pytest.raises(ValueError, match="1 vectors for 2 chunks"):
        await glue_by_similarity(
            [_chunk(0, "a", 1), _chunk(1, "b", 1)],
            _Embedder([[1.0]]),
        )


@pytest.mark.asyncio
async def test_glue_helpers_cover_cosine_awaitable_and_metadata() -> None:
    async def _vectors() -> list[list[float]]:
        return [[1.0, 2.0]]

    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([0.0], [1.0]) == 0.0
    assert await _to_list(_vectors()) == [[1.0, 2.0]]
    assert await _to_list([[3.0]]) == [[3.0]]

    window = finalize_window(
        [
            _chunk(
                0,
                "a",
                1,
                section="Intro",
                retrieval_excluded=True,
                topic_group_id="topic-1",
                slide_title="Welcome",
            ),
            _chunk(1, "b", 2, retrieval_excluded=True),
        ],
        group_id=4,
    )
    assert window.metadata["retrieval_excluded"] is True
    assert window.metadata["section"] == "Intro"
    assert window.metadata["topic_group_id"] == "topic-1"


def test_absorb_tiny_windows_respects_role_and_size() -> None:
    unchanged = absorb_tiny_windows(
        [_chunk(0, "tiny", 1, role="summary"), _chunk(1, "body", 10)],
        min_tokens=5,
        max_tokens=20,
    )
    assert len(unchanged) == 2

    too_large = absorb_tiny_windows(
        [_chunk(0, "tiny", 1), _chunk(1, "body", 20)],
        min_tokens=5,
        max_tokens=20,
    )
    assert len(too_large) == 2


@pytest.mark.asyncio
async def test_prune_superseded_graph_returns_removed_count() -> None:
    first_result = SimpleNamespace(single=AsyncMock(return_value={"removed": 3}))
    session = SimpleNamespace(run=AsyncMock(side_effect=[first_result, object()]))
    client = SimpleNamespace(session=lambda: _SessionContext(session))

    removed = await prune_superseded_chunk_graph(
        client,
        material_id=uuid4(),
        material_version_id=uuid4(),
        org_id=uuid4(),
    )

    assert removed == 3
    assert session.run.await_count == 2


@pytest.mark.asyncio
async def test_prune_superseded_graph_is_best_effort() -> None:
    session = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("neo4j down")))
    client = SimpleNamespace(session=lambda: _SessionContext(session))

    assert (
        await prune_superseded_chunk_graph(
            client,
            material_id=uuid4(),
            material_version_id=uuid4(),
            org_id=uuid4(),
        )
        == 0
    )


@pytest.mark.asyncio
async def test_bm25_guards_invalid_queries_without_database_call() -> None:
    db = SimpleNamespace(execute=AsyncMock())
    assert await bm25_search(db, "query", top_k=0) == []
    assert await bm25_search(db, "   ") == []
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_bm25_maps_database_rows_and_normalizes_filters() -> None:
    ids = [uuid4() for _ in range(4)]
    rows = [
        {
            "id": ids[0],
            "material_version_id": ids[1],
            "course_id": ids[2],
            "lesson_id": ids[3],
            "content": "normal forms",
            "rank": Decimal("0.75"),
            "metadata": ["not", "a", "mapping"],
        }
    ]
    mappings = SimpleNamespace(all=lambda: rows)
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(mappings=lambda: mappings)))

    found = await bm25_search(
        db,
        "normal forms",
        course_id=ids[2],
        lesson_ids=[ids[3]],
        top_k=7,
    )

    assert found[0].rank == 0.75
    assert found[0].metadata is None
    params = db.execute.await_args.args[1]
    assert params["lesson_ids"] == [ids[3]]
    assert params["top_k"] == 7


@pytest.mark.asyncio
async def test_readiness_queries_map_rows_and_persist_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    student_id, path_id, version_id = uuid4(), uuid4(), uuid4()
    db = SimpleNamespace(add=lambda value: setattr(db, "added", value), flush=AsyncMock())
    snapshot = await readiness.insert_snapshot(
        db,
        student_id=student_id,
        career_path_id=path_id,
        version_id=version_id,
        readiness_score=Decimal("82.50"),
    )
    assert db.added is snapshot
    db.flush.assert_awaited_once()

    rows = [SimpleNamespace(student_id=student_id, career_path_id=path_id, version_id=version_id)]
    result = SimpleNamespace(all=lambda: rows)
    db.execute = AsyncMock(return_value=result)
    assert await readiness.list_active_enrollment_pairs(db) == [
        (student_id, path_id, version_id)
    ]


@pytest.mark.asyncio
async def test_readiness_history_and_latest_projection() -> None:
    student_id, path_id = uuid4(), uuid4()
    snapshots = [object(), object()]
    scalar_result = SimpleNamespace(all=lambda: snapshots)
    history_result = SimpleNamespace(scalars=lambda: scalar_result)
    captured_at = object()
    latest_result = SimpleNamespace(
        all=lambda: [
            SimpleNamespace(
                student_id=student_id,
                primary_email="student@example.com",
                readiness_score=Decimal("91.0"),
                captured_at=captured_at,
            )
        ]
    )
    db = SimpleNamespace(execute=AsyncMock(side_effect=[history_result, latest_result]))

    assert await readiness.list_my_snapshots(
        db, student_id=student_id, career_path_id=path_id, limit=5
    ) == snapshots
    assert await readiness.latest_snapshots_for_path(db, path_id) == [
        {
            "student_id": student_id,
            "student_email": "student@example.com",
            "readiness_score": Decimal("91.0"),
            "captured_at": captured_at,
        }
    ]
