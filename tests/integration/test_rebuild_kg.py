"""KG-rebuild script integration tests (T9.2).

Mocks the KG builder + Neo4j count so no LLM / Neo4j traffic happens; uses a
real Postgres DB (per the ``test_ingestion_pipeline`` pattern) so the
discovery query and cost-tracking aggregation are exercised against actual
SQL. Each test scopes to its seeded version IDs by monkey-patching
``_discover_candidates`` so leftover ready rows from neighbouring tests
cannot leak in.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.features.courses.models  # noqa: F401  -- register course/module/lesson FK targets
import abridgeai.features.identity.models  # noqa: F401  -- register users + storage_objects FK targets
from abridgeai.ai.llm.audit import write_ai_model_call
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import get_settings
from scripts import rebuild_knowledge_graph as rebuild_kg


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
    command.upgrade(cfg, "head")


async def _reflect_audit_parent_tables(eng: AsyncEngine) -> None:
    from abridgeai.core.db import Base

    needed = {"generation_runs"}
    missing = needed - set(Base.metadata.tables)
    if not missing:
        return
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.reflect(bind=sync_conn, only=tuple(missing))
        )


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    await _reflect_audit_parent_tables(eng)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@dataclass
class _Scope:
    org_id: UUID
    owner_id: UUID
    course_id: UUID
    module_id: UUID
    lesson_id: UUID
    storage_id: UUID
    material_id: UUID
    version_ids: list[UUID]
    processing_job_id: UUID


async def _seed_versions(engine: AsyncEngine, *, count: int) -> _Scope:
    scope = _Scope(
        org_id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        course_id=uuid.uuid4(),
        module_id=uuid.uuid4(),
        lesson_id=uuid.uuid4(),
        storage_id=uuid.uuid4(),
        material_id=uuid.uuid4(),
        version_ids=[uuid.uuid4() for _ in range(count)],
        processing_job_id=uuid.uuid4(),
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) VALUES (:id, :s, :n, 'active')"
            ),
            {"id": scope.org_id, "s": f"kgr-{scope.org_id.hex[:8]}", "n": "KGR Org"},
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email, status) VALUES (:id, :e, 'active')"),
            {"id": scope.owner_id, "e": f"kgr-{scope.owner_id.hex[:8]}@test.local"},
        )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :o, :u, :s, 'KGR Course', 'draft')"
            ),
            {
                "id": scope.course_id,
                "o": scope.org_id,
                "u": scope.owner_id,
                "s": f"kgr-c-{scope.course_id.hex[:6]}",
            },
        )
        await conn.execute(
            text("INSERT INTO modules (id, course_id, title, position) VALUES (:id, :c, 'Mod', 1)"),
            {"id": scope.module_id, "c": scope.course_id},
        )
        await conn.execute(
            text("INSERT INTO lessons (id, module_id, slug, title) VALUES (:id, :m, :s, 'Lsn')"),
            {
                "id": scope.lesson_id,
                "m": scope.module_id,
                "s": f"kgr-l-{scope.lesson_id.hex[:6]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, 'test-bucket', :key, 'text/plain')"
            ),
            {
                "id": scope.storage_id,
                "key": f"kgr/{scope.storage_id.hex}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:id, :l, 'Test Mat', 'text')"
            ),
            {"id": scope.material_id, "l": scope.lesson_id},
        )
        for idx, vid in enumerate(scope.version_ids):
            await conn.execute(
                text(
                    "INSERT INTO learning_material_versions "
                    "(id, material_id, storage_object_id, version_no, processing_status) "
                    "VALUES (:id, :m, :so, :v, 'ready')"
                ),
                {
                    "id": vid,
                    "m": scope.material_id,
                    "so": scope.storage_id,
                    "v": idx + 1,
                },
            )
        # Parent processing_job for the cost-attribution audit rows the
        # budget test writes (ck_ai_model_calls_parent_ref demands one of
        # processing_job_id / generation_run_id be NOT NULL).
        await conn.execute(
            text(
                "INSERT INTO processing_jobs "
                "(id, entity_type, entity_id, job_type, status) "
                "VALUES (:id, 'material_version', :ent, 'full_pipeline', 'completed')"
            ),
            {"id": scope.processing_job_id, "ent": scope.version_ids[0]},
        )
    return scope


async def _teardown_scope(engine: AsyncEngine, scope: _Scope) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM ai_model_calls WHERE model_name = 'fake-kgr-model'"))
        await conn.execute(
            text("DELETE FROM processing_jobs WHERE id = :id"),
            {"id": scope.processing_job_id},
        )
        await conn.execute(
            text("DELETE FROM document_chunks WHERE material_version_id = ANY(:ids)"),
            {"ids": list(scope.version_ids)},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = ANY(:ids)"),
            {"ids": list(scope.version_ids)},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :id"),
            {"id": scope.material_id},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": scope.storage_id},
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :id"), {"id": scope.lesson_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :id"), {"id": scope.module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": scope.course_id})
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": scope.owner_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": scope.org_id})


@asynccontextmanager
async def _scope_with(engine: AsyncEngine, *, count: int) -> AsyncIterator[_Scope]:
    scope = await _seed_versions(engine, count=count)
    try:
        yield scope
    finally:
        await _teardown_scope(engine, scope)


def _scope_discover(
    monkeypatch: pytest.MonkeyPatch,
    scope: _Scope,
    *,
    max_materials: int | None = None,
) -> None:
    """Filter ``_discover_candidates`` to only return rows from this test's scope.

    Real query semantics (ORDER BY uploaded_at DESC, LIMIT) are preserved by
    filtering the real result on the seeded version_ids. This keeps the
    SQL path live while making the test hermetic against unrelated ``ready``
    rows that earlier tests may have left behind.
    """
    real = rebuild_kg._discover_candidates
    seeded = set(scope.version_ids)

    async def _wrapped(
        db: AsyncSession,
        args: rebuild_kg.RebuildArgs,
    ) -> list[rebuild_kg._Candidate]:
        all_rows = await real(db, args)
        filtered = [c for c in all_rows if c.version_id in seeded]
        if max_materials is not None:
            filtered = filtered[:max_materials]
        return filtered

    monkeypatch.setattr(rebuild_kg, "_discover_candidates", _wrapped)


class _FakeKGClient:
    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_dry_run_lists_versions_no_writes(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with _scope_with(engine, count=3) as scope:
        _scope_discover(monkeypatch, scope)
        builder_calls: list[UUID] = []

        async def _builder(version_id: UUID, *args: Any, **kwargs: Any) -> Any:
            builder_calls.append(version_id)

        count_calls: list[int] = []

        async def _count_fn() -> int:
            count_calls.append(0)
            return 10

        args = rebuild_kg.RebuildArgs(
            workers=1,
            max_materials=None,
            dry_run=True,
            since=None,
            material_id=None,
            budget_usd=None,
        )

        exit_code, report = await rebuild_kg.run(
            args,
            sessionmaker=session_factory,
            builder=_builder,
            kg_client_factory=lambda: _FakeKGClient(),
            concept_count_fn=_count_fn,
        )

    assert exit_code == rebuild_kg.EXIT_OK
    assert report.dry_run is True
    assert report.total_candidates == 3
    assert builder_calls == []
    assert count_calls == []  # Neo4j count never queried in dry-run.

    captured = capsys.readouterr().out
    assert "DRY RUN" in captured
    for vid in scope.version_ids:
        assert str(vid) in captured


@pytest.mark.asyncio
async def test_max_materials_limit(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _scope_with(engine, count=5) as scope:
        _scope_discover(monkeypatch, scope, max_materials=2)
        builder_calls: list[UUID] = []

        async def _builder(version_id: UUID, *args: Any, **kwargs: Any) -> Any:
            builder_calls.append(version_id)

        async def _count_fn() -> int:
            return 0

        args = rebuild_kg.RebuildArgs(
            workers=1,
            max_materials=2,
            dry_run=False,
            since=None,
            material_id=None,
            budget_usd=None,
        )

        exit_code, report = await rebuild_kg.run(
            args,
            sessionmaker=session_factory,
            builder=_builder,
            kg_client_factory=lambda: _FakeKGClient(),
            concept_count_fn=_count_fn,
        )

    assert exit_code == rebuild_kg.EXIT_OK
    assert report.total_candidates == 2
    assert report.processed == 2
    assert report.failed == 0
    assert len(builder_calls) == 2
    assert all(vid in scope.version_ids for vid in builder_calls)


@pytest.mark.asyncio
async def test_failure_continues_to_next(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _scope_with(engine, count=3) as scope:
        _scope_discover(monkeypatch, scope)
        attempted: list[UUID] = []
        failing_version_id: UUID | None = None

        async def _builder(version_id: UUID, *args: Any, **kwargs: Any) -> Any:
            attempted.append(version_id)
            nonlocal failing_version_id
            if failing_version_id is None:
                failing_version_id = version_id
                raise RuntimeError("synthetic failure: KG extraction blew up")

        async def _count_fn() -> int:
            return 5

        args = rebuild_kg.RebuildArgs(
            workers=1,
            max_materials=None,
            dry_run=False,
            since=None,
            material_id=None,
            budget_usd=None,
        )

        exit_code, report = await rebuild_kg.run(
            args,
            sessionmaker=session_factory,
            builder=_builder,
            kg_client_factory=lambda: _FakeKGClient(),
            concept_count_fn=_count_fn,
        )

    assert exit_code == rebuild_kg.EXIT_OK
    assert report.total_candidates == 3
    assert report.processed == 2
    assert report.failed == 1
    assert len(attempted) == 3
    assert failing_version_id is not None
    assert failing_version_id in report.failed_ids


@pytest.mark.asyncio
async def test_budget_aborts_when_exceeded(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _scope_with(engine, count=3) as scope:
        _scope_discover(monkeypatch, scope)
        builder_calls: list[UUID] = []

        async def _builder(version_id: UUID, *args: Any, **kwargs: Any) -> Any:
            db: AsyncSession = kwargs["db"]
            run_id = kwargs["pipeline_run_id"]
            await write_ai_model_call(
                db,
                role=LLMRole.KG_EXTRACTION,
                tier="small",
                operation="chat_completion",
                model_name="fake-kgr-model",
                base_url="https://fake.test/v1",
                stage_name="kg_build",
                pipeline_run_id=run_id,
                parent_run_id=None,
                parent_job_id=scope.processing_job_id,
                request_payload={},
                response_payload={},
                input_tokens=10,
                output_tokens=5,
                cached_input_tokens=None,
                latency_ms=1,
                status="success",
                error_message=None,
                estimated_cost_usd=Decimal("5.00"),
            )
            builder_calls.append(version_id)

        async def _count_fn() -> int:
            return 0

        args = rebuild_kg.RebuildArgs(
            workers=1,
            max_materials=None,
            dry_run=False,
            since=None,
            material_id=None,
            budget_usd=Decimal("3.00"),
        )

        exit_code, report = await rebuild_kg.run(
            args,
            sessionmaker=session_factory,
            builder=_builder,
            kg_client_factory=lambda: _FakeKGClient(),
            concept_count_fn=_count_fn,
        )

    assert exit_code == rebuild_kg.EXIT_BUDGET_EXCEEDED
    assert report.budget_exceeded is True
    # First version always processes; subsequent ones are short-circuited by
    # the budget guard. Asserts at least one ran but not all three.
    assert report.processed >= 1
    assert report.processed < 3
    assert len(builder_calls) == report.processed


@pytest.mark.asyncio
async def test_post_count_within_5_percent_of_pre_count(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _scope_with(engine, count=3) as scope:
        _scope_discover(monkeypatch, scope)

        async def _builder(version_id: UUID, *args: Any, **kwargs: Any) -> Any:
            return None

        # Pre-count then post-count (+2% delta — inside ±5% tolerance).
        counts = iter([100, 102])

        async def _count_fn() -> int:
            return next(counts)

        args = rebuild_kg.RebuildArgs(
            workers=2,
            max_materials=None,
            dry_run=False,
            since=None,
            material_id=None,
            budget_usd=None,
        )

        exit_code, report = await rebuild_kg.run(
            args,
            sessionmaker=session_factory,
            builder=_builder,
            kg_client_factory=lambda: _FakeKGClient(),
            concept_count_fn=_count_fn,
        )

    assert exit_code == rebuild_kg.EXIT_OK
    assert report.pre_count == 100
    assert report.post_count == 102
    assert report.delta_pct is not None
    assert abs(report.delta_pct) <= rebuild_kg.DELTA_TOLERANCE_PCT
    assert report.processed == 3


@pytest.mark.asyncio
async def test_delta_out_of_tolerance_returns_warning_exit_code(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _scope_with(engine, count=2) as scope:
        _scope_discover(monkeypatch, scope)

        async def _builder(version_id: UUID, *args: Any, **kwargs: Any) -> Any:
            return None

        # +100% — well past ±5% tolerance.
        counts = iter([100, 200])

        async def _count_fn() -> int:
            return next(counts)

        args = rebuild_kg.RebuildArgs(
            workers=1,
            max_materials=None,
            dry_run=False,
            since=None,
            material_id=None,
            budget_usd=None,
        )

        exit_code, report = await rebuild_kg.run(
            args,
            sessionmaker=session_factory,
            builder=_builder,
            kg_client_factory=lambda: _FakeKGClient(),
            concept_count_fn=_count_fn,
        )

    assert exit_code == rebuild_kg.EXIT_DELTA_OUT_OF_TOLERANCE
    assert report.delta_pct is not None
    assert report.delta_pct == 100.0
