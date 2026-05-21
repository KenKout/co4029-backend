"""Ad-hoc reprocess script for Phase 2 contextual embeddings rollout.

Runs the materials reprocess flow end-to-end without going through the
HTTP layer (which requires OAuth). Useful for re-embedding a single
material after a chunker/embedding-input change like Phase 2 of the
contextual retrieval upgrade.

Usage:
    uv run --no-sync python scripts/reprocess_material.py <material_uuid>

The script:
  1. Looks up the latest version of the material
  2. Marks it pending and clears existing chunks
  3. Enqueues an arq job for ``ingest_material_version_task``
  4. Prints the job id; the worker (pm2 abridgeai-worker) picks it up

Stage C semantic enrichment is cached on content hash, so re-runs hit
the cache and don't incur fresh LLM cost. Only the embeddings are
recomputed (cheap).

Notes / pitfalls:
  - The arq queue name is read from ``settings.arq_queue_name``.
  - Worker must be running (``pm2 status abridgeai-worker``).
  - Watch progress with: ``pm2 logs abridgeai-worker --lines 50``
  - The script exits as soon as the job is enqueued; processing
    happens asynchronously.
"""

from __future__ import annotations

import asyncio
import sys
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.features.materials.models import (
    LearningMaterial,
    LearningMaterialVersion,
)


async def main(material_id: UUID) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with SessionLocal() as db:
        material = await db.scalar(
            select(LearningMaterial).where(LearningMaterial.id == material_id)
        )
        if material is None:
            print(f"ERROR: material {material_id} not found", file=sys.stderr)
            sys.exit(1)

        version = await db.scalar(
            select(LearningMaterialVersion)
            .where(LearningMaterialVersion.material_id == material_id)
            .order_by(LearningMaterialVersion.created_at.desc())
            .limit(1)
        )
        if version is None:
            print(f"ERROR: no version for material {material_id}", file=sys.stderr)
            sys.exit(1)

        print(f"material: {material.title!r}")
        print(f"version_id: {version.id}")
        print(f"current status: {version.processing_status}")

    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    pool = await create_pool(redis_settings)
    try:
        job = await pool.enqueue_job(
            "ingest_material_version_task",
            str(version.id),
            _queue_name=settings.arq_queue_name,
        )
        if job is None:
            print("ERROR: failed to enqueue (already queued?)", file=sys.stderr)
            sys.exit(2)
        print(f"enqueued job: {job.job_id}")
    finally:
        await pool.close()
    await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/reprocess_material.py <material_uuid>",
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(main(UUID(sys.argv[1])))
