from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from abridgeai.ai.retrieval import vector_search
from abridgeai.core.config import get_settings


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _vec_literal(values: list[float], dim: int = 3072) -> str:
    padded = list(values) + [0.0] * (dim - len(values))
    return "[" + ",".join(repr(float(v)) for v in padded[:dim]) + "]"


def _vec(values: list[float], dim: int = 3072) -> list[float]:
    padded = list(values) + [0.0] * (dim - len(values))
    return [float(v) for v in padded[:dim]]


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def seeded_scope(engine: AsyncEngine):
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    course_a_id = uuid.uuid4()
    course_b_id = uuid.uuid4()
    module_a_id = uuid.uuid4()
    module_b_id = uuid.uuid4()
    lesson_ids = [uuid.uuid4() for _ in range(3)]
    storage_obj_id = uuid.uuid4()
    material_a_id = uuid.uuid4()
    material_b_id = uuid.uuid4()
    version_a_id = uuid.uuid4()
    version_b_id = uuid.uuid4()

    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {
                "id": org_id,
                "slug": f"vec-{org_id.hex[:8]}",
                "name": "Vector Search Test Org",
            },
        )
        await conn.execute(
            text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"vec-{user_id.hex[:8]}@test.local"},
        )
        for cid, slug in ((course_a_id, "course-a"), (course_b_id, "course-b")):
            await conn.execute(
                text(
                    "INSERT INTO courses (id, organization_id, owner_user_id, slug, title) "
                    "VALUES (:id, :org, :owner, :slug, :title)"
                ),
                {
                    "id": cid,
                    "org": org_id,
                    "owner": user_id,
                    "slug": f"{slug}-{cid.hex[:8]}",
                    "title": f"Course {slug}",
                },
            )
        for mid, cid, pos in ((module_a_id, course_a_id, 1), (module_b_id, course_b_id, 1)):
            await conn.execute(
                text(
                    "INSERT INTO modules (id, course_id, title, position) "
                    "VALUES (:id, :course, :title, :pos)"
                ),
                {"id": mid, "course": cid, "title": "M", "pos": pos},
            )
        for idx, lid in enumerate(lesson_ids):
            module_id = module_a_id if idx < 2 else module_b_id
            await conn.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title) "
                    "VALUES (:id, :module, :slug, :title)"
                ),
                {
                    "id": lid,
                    "module": module_id,
                    "slug": f"l-{lid.hex[:6]}",
                    "title": f"Lesson {idx}",
                },
            )
        await conn.execute(
            text(
                "INSERT INTO storage_objects (id, bucket, object_key, mime_type) "
                "VALUES (:id, :bucket, :key, :mime)"
            ),
            {
                "id": storage_obj_id,
                "bucket": "test",
                "key": f"vec/{storage_obj_id.hex}",
                "mime": "text/plain",
            },
        )
        for mat_id, lid in ((material_a_id, lesson_ids[0]), (material_b_id, lesson_ids[2])):
            await conn.execute(
                text(
                    "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                    "VALUES (:id, :lesson, :title, 'text')"
                ),
                {"id": mat_id, "lesson": lid, "title": "Material"},
            )
        for ver_id, mat_id in ((version_a_id, material_a_id), (version_b_id, material_b_id)):
            await conn.execute(
                text(
                    "INSERT INTO learning_material_versions "
                    "(id, material_id, storage_object_id, version_no, processing_status) "
                    "VALUES (:id, :mat, :obj, 1, 'ready')"
                ),
                {"id": ver_id, "mat": mat_id, "obj": storage_obj_id},
            )

    yield {
        "org_id": org_id,
        "user_id": user_id,
        "course_a_id": course_a_id,
        "course_b_id": course_b_id,
        "module_a_id": module_a_id,
        "module_b_id": module_b_id,
        "lesson_ids": lesson_ids,
        "version_a_id": version_a_id,
        "version_b_id": version_b_id,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM document_chunks WHERE course_id IN (:a, :b)"),
            {"a": course_a_id, "b": course_b_id},
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE id = ANY(:ids)"),
            {"ids": [version_a_id, version_b_id]},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = ANY(:ids)"),
            {"ids": [material_a_id, material_b_id]},
        )
        await conn.execute(
            text("DELETE FROM storage_objects WHERE id = :id"),
            {"id": storage_obj_id},
        )
        await conn.execute(
            text("DELETE FROM lessons WHERE id = ANY(:ids)"),
            {"ids": lesson_ids},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [module_a_id, module_b_id]},
        )
        await conn.execute(
            text("DELETE FROM courses WHERE id = ANY(:ids)"),
            {"ids": [course_a_id, course_b_id]},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": user_id})
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org_id})


async def _insert_chunk(
    conn,
    *,
    course_id: uuid.UUID,
    module_id: uuid.UUID,
    lesson_id: uuid.UUID,
    version_id: uuid.UUID,
    chunk_index: int,
    embedding: list[float],
    content: str,
) -> uuid.UUID:
    chunk_id = uuid.uuid4()
    await conn.execute(
        text(
            "INSERT INTO document_chunks "
            "(id, course_id, module_id, lesson_id, material_version_id, chunk_index, "
            " chunk_type, content, content_hash, embedding) "
            "VALUES (:id, :course, :module, :lesson, :version, :idx, 'text', :content, "
            "        :hash, CAST(:embedding AS halfvec))"
        ),
        {
            "id": chunk_id,
            "course": course_id,
            "module": module_id,
            "lesson": lesson_id,
            "version": version_id,
            "idx": chunk_index,
            "content": content,
            "hash": chunk_id.hex,
            "embedding": _vec_literal(embedding),
        },
    )
    return chunk_id


async def test_vector_search_returns_chunks_in_distance_order(engine: AsyncEngine, seeded_scope):
    chunk_ids: list[uuid.UUID] = []
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ]
    async with engine.begin() as conn:
        for i, emb in enumerate(embeddings):
            cid = await _insert_chunk(
                conn,
                course_id=seeded_scope["course_a_id"],
                module_id=seeded_scope["module_a_id"],
                lesson_id=seeded_scope["lesson_ids"][0],
                version_id=seeded_scope["version_a_id"],
                chunk_index=i,
                embedding=emb,
                content=f"chunk-{i}",
            )
            chunk_ids.append(cid)

    target_idx = 2
    query_emb = _vec(embeddings[target_idx])

    async with AsyncSession(engine) as session:
        results = await vector_search(session, query_emb, top_k=5)

    assert len(results) == 5
    assert results[0].chunk_id == chunk_ids[target_idx]
    assert results[0].distance == pytest.approx(0.0, abs=1e-6)
    distances = [r.distance for r in results]
    assert distances == sorted(distances)


async def test_vector_search_filters_by_course_id(engine: AsyncEngine, seeded_scope):
    async with engine.begin() as conn:
        await _insert_chunk(
            conn,
            course_id=seeded_scope["course_a_id"],
            module_id=seeded_scope["module_a_id"],
            lesson_id=seeded_scope["lesson_ids"][0],
            version_id=seeded_scope["version_a_id"],
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0, 0.0],
            content="course-A",
        )
        await _insert_chunk(
            conn,
            course_id=seeded_scope["course_b_id"],
            module_id=seeded_scope["module_b_id"],
            lesson_id=seeded_scope["lesson_ids"][2],
            version_id=seeded_scope["version_b_id"],
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0, 0.0],
            content="course-B",
        )

    async with AsyncSession(engine) as session:
        results = await vector_search(
            session,
            _vec([1.0, 0.0, 0.0, 0.0]),
            course_id=seeded_scope["course_a_id"],
            top_k=10,
        )

    assert len(results) == 1
    assert results[0].course_id == seeded_scope["course_a_id"]
    assert results[0].content == "course-A"


async def test_vector_search_filters_by_lesson_ids(engine: AsyncEngine, seeded_scope):
    lesson_ids = seeded_scope["lesson_ids"]
    async with engine.begin() as conn:
        await _insert_chunk(
            conn,
            course_id=seeded_scope["course_a_id"],
            module_id=seeded_scope["module_a_id"],
            lesson_id=lesson_ids[0],
            version_id=seeded_scope["version_a_id"],
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0, 0.0],
            content="L0",
        )
        await _insert_chunk(
            conn,
            course_id=seeded_scope["course_a_id"],
            module_id=seeded_scope["module_a_id"],
            lesson_id=lesson_ids[1],
            version_id=seeded_scope["version_a_id"],
            chunk_index=1,
            embedding=[1.0, 0.0, 0.0, 0.0],
            content="L1",
        )
        await _insert_chunk(
            conn,
            course_id=seeded_scope["course_b_id"],
            module_id=seeded_scope["module_b_id"],
            lesson_id=lesson_ids[2],
            version_id=seeded_scope["version_b_id"],
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0, 0.0],
            content="L2",
        )

    async with AsyncSession(engine) as session:
        results = await vector_search(
            session,
            _vec([1.0, 0.0, 0.0, 0.0]),
            lesson_ids=[lesson_ids[0], lesson_ids[1]],
            top_k=10,
        )

    contents = sorted(r.content for r in results)
    assert contents == ["L0", "L1"]


async def test_vector_search_top_k_respected(engine: AsyncEngine, seeded_scope):
    n = 25
    async with engine.begin() as conn:
        for i in range(n):
            tilt = i / float(n)
            await _insert_chunk(
                conn,
                course_id=seeded_scope["course_a_id"],
                module_id=seeded_scope["module_a_id"],
                lesson_id=seeded_scope["lesson_ids"][0],
                version_id=seeded_scope["version_a_id"],
                chunk_index=i,
                embedding=[1.0 - tilt, tilt, 0.0, 0.0],
                content=f"c-{i}",
            )

    async with AsyncSession(engine) as session:
        results = await vector_search(
            session,
            _vec([1.0, 0.0, 0.0, 0.0]),
            course_id=seeded_scope["course_a_id"],
            top_k=5,
        )

    assert len(results) == 5


async def test_vector_search_includes_embeddings_when_requested(engine: AsyncEngine, seeded_scope):
    async with engine.begin() as conn:
        await _insert_chunk(
            conn,
            course_id=seeded_scope["course_a_id"],
            module_id=seeded_scope["module_a_id"],
            lesson_id=seeded_scope["lesson_ids"][0],
            version_id=seeded_scope["version_a_id"],
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0, 0.0],
            content="with-emb",
        )

    async with AsyncSession(engine) as session:
        without = await vector_search(
            session,
            _vec([1.0, 0.0, 0.0, 0.0]),
            course_id=seeded_scope["course_a_id"],
            top_k=1,
        )
        with_emb = await vector_search(
            session,
            _vec([1.0, 0.0, 0.0, 0.0]),
            course_id=seeded_scope["course_a_id"],
            top_k=1,
            include_embeddings=True,
        )

    assert without[0].embedding is None
    assert with_emb[0].embedding is not None
    assert len(with_emb[0].embedding) == 3072
    assert with_emb[0].embedding[0] == pytest.approx(1.0, abs=1e-6)


async def test_vector_search_top_k_zero_returns_empty(engine: AsyncEngine, seeded_scope):
    async with AsyncSession(engine) as session:
        results = await vector_search(session, _vec([1.0, 0.0, 0.0, 0.0]), top_k=0)
    assert results == []
