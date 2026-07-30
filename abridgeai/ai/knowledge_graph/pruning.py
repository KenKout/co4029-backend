"""Superseded-chunk pruning for the knowledge graph.

Split out of ``builder.py`` (which carries the LLM extraction + upsert path)
to keep both under the 500-LOC module budget enforced by
``test_no_god_files_under_abridgeai_ai``.
"""

from __future__ import annotations

import logging
from uuid import UUID

from abridgeai.infrastructure.neo4j import KnowledgeGraphClient

logger = logging.getLogger(__name__)


async def prune_superseded_chunk_graph(
    client: KnowledgeGraphClient,
    *,
    material_id: UUID,
    material_version_id: UUID,
    org_id: UUID,
) -> int:
    """Delete ``Chunk`` nodes under this Material left by EARLIER versions.

    ``_persist_chunks`` deletes and recreates every ``document_chunks`` row on
    each ingest, so a chunk's Postgres UUID — and therefore ``Chunk.id`` — is
    new every run. Nothing ever removed the old graph nodes, so each reprocess
    layered another generation of ``Chunk`` nodes under the same ``Material``,
    each still carrying ``MENTIONS_CONCEPT`` edges to concepts extracted from
    text that no longer exists. Retrieval walks
    ``Lesson->Material->Chunk->Concept``, so those stale mentions were being
    served as current course content.

    Only *superseded* versions are pruned. Chunks belonging to
    ``material_version_id`` are left alone, because that is exactly what
    ``_already_built_previews_by_index`` resumes from — a build killed by
    ``job_timeout`` must still be able to continue where it stopped.

    Nodes written before ``material_version_id`` was stamped have the property
    as NULL and are un-attributable; they are pruned too, which costs one
    full rebuild the first time this runs and leaves the graph consistent
    afterwards.

    Best-effort: never raises. A failed prune leaves stale nodes (the status
    quo), which must not be allowed to fail an ingest.
    """
    try:
        async with client.session() as session:
            result = await session.run(
                """
                MATCH (m:Material {id: $material_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.material_version_id IS NULL
                   OR c.material_version_id <> $material_version_id
                DETACH DELETE c
                RETURN count(c) AS removed
                """,
                material_id=str(material_id),
                material_version_id=str(material_version_id),
            )
            record = await result.single()
            removed = int(record["removed"]) if record else 0

            # A Concept with no remaining mention is unreachable from any
            # chunk and can only pollute name-based anchor lookups. Scoped to
            # the tenant so one course's reprocess can never touch another's.
            await session.run(
                """
                MATCH (concept:Concept {org_id: $org_id})
                WHERE NOT (concept)<-[:MENTIONS_CONCEPT]-()
                DETACH DELETE concept
                """,
                org_id=str(org_id),
            )
    except Exception:  # noqa: BLE001 -- pruning is hygiene; never fail an ingest
        logger.warning(
            "kg_build prune failed for material_version %s; stale nodes retained",
            material_version_id,
            exc_info=True,
        )
        return 0
    if removed:
        logger.info(
            "kg_build pruned %d superseded chunk nodes for material_version %s",
            removed,
            material_version_id,
        )
    return removed


__all__ = ["prune_superseded_chunk_graph"]
