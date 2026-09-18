"""Knowledge-graph links reused by cloned lessons."""

from __future__ import annotations

import logging
from uuid import UUID

from abridgeai.core.config import get_settings
from abridgeai.infrastructure.neo4j import graph_client

logger = logging.getLogger(__name__)


async def clone_lesson_material_link(
    *, source_lesson_id: UUID, target_lesson_id: UUID, target_title: str
) -> bool:
    """Attach a cloned lesson to the source lesson's existing material graph.

    Course cloning creates a new lesson id but intentionally reuses the source
    material. Neo4j keys the ``Lesson`` node separately from ``Material``, so
    the clone needs its own ``Lesson -[:HAS_MATERIAL]-> Material`` edge before
    graph retrieval can see the existing concept/chunk graph.
    """
    if not get_settings().knowledge_graph_enabled:
        return False
    try:
        async with graph_client() as client, client.session() as session:
            result = await session.run(
                """
                MATCH (source:Lesson {id: $source_lesson_id})
                      -[:HAS_MATERIAL]->(material:Material)
                MERGE (target:Lesson {id: $target_lesson_id})
                  SET target.title = $target_title
                MERGE (target)-[:HAS_MATERIAL]->(material)
                RETURN count(material) AS linked
                """,
                source_lesson_id=str(source_lesson_id),
                target_lesson_id=str(target_lesson_id),
                target_title=target_title,
            )
            record = await result.single()
            return bool(record and record["linked"])
    except Exception:  # noqa: BLE001 -- clone should not block course duplication
        logger.warning("clone lesson KG link failed", exc_info=True)
        return False


__all__ = ["clone_lesson_material_link"]
