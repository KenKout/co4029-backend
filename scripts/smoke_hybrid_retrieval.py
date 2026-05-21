"""Smoke test for Phase 3 hybrid retrieval against live Postgres.

Runs bm25_search + reciprocal_rank_fusion over the existing 63 chunks
in document_chunks. Requires the backend .env to be loaded (DATABASE_URL).
"""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

# Ensure the project root is importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from abridgeai.ai.retrieval import bm25_search
from abridgeai.ai.retrieval.fusion import reciprocal_rank_fusion
from abridgeai.ai.retrieval.pgvector import ChunkWithDistance


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        # Fallback: assume default local docker setup.
        db_url = (
            "postgresql+psycopg://abridgeai:"
            + os.environ.get("POSTGRES_PASSWORD", "abridgeai")
            + "@localhost:5432/abridgeai"
        )

    engine = create_async_engine(db_url, future=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async with Session() as db:
        # 1) BM25 only — verify the retriever surfaces relevant chunks
        bm25_hits = await bm25_search(db, "data warehouse architecture", top_k=10)
        print(f"BM25 query 'data warehouse architecture': {len(bm25_hits)} hits")
        for hit in bm25_hits[:3]:
            preview = hit.content[:80].replace("\n", " ")
            print(f"  rank={hit.rank:.4f} chunk={hit.chunk_id}  {preview!r}")

        # 2) Synthetic semantic hits — fake a vector_search result that
        # ranks DIFFERENT chunks first, then verify RRF promotes a
        # cross-source overlap chunk to the top.
        if not bm25_hits:
            print("No BM25 hits — skipping fusion smoke")
            return

        common_id = bm25_hits[0].chunk_id
        unique_id = uuid4()
        synthetic_semantic = [
            ChunkWithDistance(
                chunk_id=unique_id,
                material_version_id=uuid4(),
                course_id=None,
                lesson_id=None,
                content="synthetic-semantic-only",
                distance=0.05,
            ),
            ChunkWithDistance(
                chunk_id=common_id,
                material_version_id=bm25_hits[0].material_version_id,
                course_id=bm25_hits[0].course_id,
                lesson_id=bm25_hits[0].lesson_id,
                content=bm25_hits[0].content,
                distance=0.10,
            ),
        ]
        fused = reciprocal_rank_fusion(
            synthetic_semantic,
            bm25_hits[:5],
            semantic_weight=0.8,
            bm25_weight=0.2,
        )
        print(f"\nRRF fused list (top 5 of {len(fused)}):")
        for f in fused[:5]:
            preview = f.content[:60].replace("\n", " ")
            print(
                f"  score={f.fused_score:.4f} sources={sorted(f.sources)} "
                f"chunk={f.chunk_id}  {preview!r}"
            )

        # 3) Sanity check: dual-source chunk should outrank
        # semantic-only and bm25-only singletons.
        top = fused[0]
        if top.chunk_id == common_id and top.sources == frozenset({"vector", "bm25"}):
            print("\nOK — dual-source chunk surfaced first as expected.")
        else:
            print(f"\nWARN — top fused chunk is {top.chunk_id} sources={top.sources}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
