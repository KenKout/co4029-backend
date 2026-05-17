from __future__ import annotations

import os

import psycopg


def _sync_url() -> str:
    url = os.environ.get(
        "DATABASE_URL", "postgresql://abridgeai:abridgeai@localhost:5433/abridgeai"
    )
    return url.replace("+psycopg_async", "").replace("+psycopg", "")


def _query(sql: str) -> list[tuple]:
    with psycopg.connect(_sync_url()) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def test_pgvector_column_type() -> None:
    rows = _query(
        "SELECT data_type, udt_name FROM information_schema.columns "
        "WHERE table_name='document_chunks' AND column_name='embedding'"
    )
    assert rows, "embedding column missing"
    data_type, udt_name = rows[0]
    assert data_type == "USER-DEFINED"
    assert udt_name == "vector"


def test_document_chunks_denormalized_fks_in_db() -> None:
    rows = _query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='document_chunks' "
        "AND column_name IN ('course_id','module_id','lesson_id') "
        "ORDER BY column_name"
    )
    names = [r[0] for r in rows]
    assert names == ["course_id", "lesson_id", "module_id"]


def test_processing_status_check_includes_enriching_in_db() -> None:
    rows = _query(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = 'learning_material_versions'::regclass "
        "AND contype = 'c' "
        "AND conname = 'learning_material_versions_processing_status_check'"
    )
    assert rows
    src = rows[0][0]
    for value in (
        "pending",
        "extracting",
        "chunking",
        "enriching",
        "embedding",
        "building_kg",
        "ready",
        "failed",
        "cancelled",
    ):
        assert f"'{value}'" in src, f"{value!r} missing from {src!r}"


def test_chunking_enrichment_cache_unique_in_db() -> None:
    rows = _query(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = 'chunking_enrichment_cache'::regclass "
        "AND conname = 'uq_chunking_enrichment_cache_hash_prompt'"
    )
    assert rows, "unique constraint missing"
    src = rows[0][0]
    assert "content_hash" in src
    assert "prompt_version" in src
