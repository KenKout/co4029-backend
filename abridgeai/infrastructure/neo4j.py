"""Neo4j async driver wrapper. Driver lifecycle only — high-level KG logic
moves to ``abridgeai/ai/knowledge_graph/`` in T2.7."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncManagedTransaction

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import AppError


class KnowledgeGraphDisabledError(AppError):
    """Raised when Neo4j is requested but ``knowledge_graph_enabled`` is False
    (or required credentials are missing). Distinct from a connection error:
    this means the feature is intentionally off, not that Neo4j is down."""


_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    global _driver
    settings = get_settings()
    if not settings.knowledge_graph_enabled:
        raise KnowledgeGraphDisabledError("Knowledge graph is not enabled")
    if not settings.neo4j_password:
        raise KnowledgeGraphDisabledError(
            "NEO4J_PASSWORD is required when knowledge graph is enabled"
        )
    if not settings.neo4j_uri:
        raise KnowledgeGraphDisabledError("NEO4J_URI is required when knowledge graph is enabled")
    if _driver is None:
        password = settings.neo4j_password.get_secret_value()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        )
    return _driver


async def close_neo4j() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
    _driver = None


def _reset_for_tests() -> None:
    """Test hook: drop cached driver without awaiting close."""
    global _driver
    _driver = None


@asynccontextmanager
async def graph_client() -> AsyncIterator[KnowledgeGraphClient]:
    yield KnowledgeGraphClient(get_neo4j_driver())


class KnowledgeGraphClient:
    def __init__(self, driver: AsyncDriver) -> None:
        self.driver = driver

    async def upsert_chunk_graph(
        self,
        *,
        hierarchy: dict[str, Any],
        chunk: dict[str, Any],
        concepts: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        async with self.driver.session() as session:
            await session.execute_write(
                _upsert_chunk_graph_tx,
                hierarchy,
                chunk,
                concepts,
                relationships,
            )

    async def lesson_concepts(self, lesson_id: str) -> list[dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (:Lesson {id: $lesson_id})-[:HAS_MATERIAL]->(:Material)
                  -[:HAS_CHUNK]->(:Chunk)-[mention:MENTIONS_CONCEPT]->(concept:Concept)
                RETURN concept.name AS name,
                       concept.type AS type,
                       concept.definition AS definition,
                       max(coalesce(mention.confidence, 0.0)) AS confidence
                ORDER BY confidence DESC, name ASC
                """,
                lesson_id=lesson_id,
            )
            return [dict(record) async for record in result]

    async def lesson_concept_graph(
        self, lesson_id: str, *, depth: int = 2
    ) -> dict[str, list[dict[str, Any]]]:
        depth = max(1, min(depth, 3))
        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH (:Lesson {{id: $lesson_id}})-[:HAS_MATERIAL]->(:Material)
                  -[:HAS_CHUNK]->(:Chunk)-[:MENTIONS_CONCEPT]->(seed:Concept)
                OPTIONAL MATCH path = (seed)-[:RELATED_TO|PREREQUISITE_OF*1..{depth}]-(related:Concept)
                WITH collect(DISTINCT seed) + collect(DISTINCT related) AS node_list, collect(path) AS paths
                UNWIND node_list AS node
                WITH collect(DISTINCT node) AS nodes, paths
                UNWIND paths AS path
                UNWIND relationships(path) AS rel
                RETURN [node IN nodes WHERE node IS NOT NULL | {{
                    id: node.name_norm,
                    label: node.name,
                    type: node.type,
                    definition: node.definition
                }}] AS nodes,
                collect(DISTINCT {{
                    source: startNode(rel).name_norm,
                    target: endNode(rel).name_norm,
                    relation: coalesce(rel.relation, type(rel)),
                    evidence: rel.evidence,
                    confidence: rel.confidence
                }}) AS edges
                """,  # noqa: E501
                lesson_id=lesson_id,
            )
            record = await result.single()
            if record is None:
                return {"nodes": [], "edges": []}
            return {"nodes": record["nodes"] or [], "edges": record["edges"] or []}


async def _upsert_chunk_graph_tx(
    tx: AsyncManagedTransaction,
    hierarchy: dict[str, Any],
    chunk: dict[str, Any],
    concepts: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    await tx.run(
        """
        MERGE (course:Course {id: $course_id})
          SET course.title = $course_title
        MERGE (module:Module {id: $module_id})
          SET module.title = $module_title
        MERGE (lesson:Lesson {id: $lesson_id})
          SET lesson.title = $lesson_title
        MERGE (material:Material {id: $material_id})
          SET material.title = $material_title, material.type = $material_type
        MERGE (chunk:Chunk {id: $chunk_id})
          SET chunk.index = $chunk_index, chunk.text_preview = $text_preview
        MERGE (course)-[:CONTAINS_MODULE]->(module)
        MERGE (module)-[:CONTAINS_LESSON]->(lesson)
        MERGE (lesson)-[:HAS_MATERIAL]->(material)
        MERGE (material)-[:HAS_CHUNK]->(chunk)
        WITH chunk
        UNWIND $concepts AS concept_payload
        MERGE (concept:Concept {name_norm: toLower(concept_payload.name)})
          SET concept.name = concept_payload.name,
              concept.type = coalesce(concept_payload.type, 'Concept'),
              concept.definition = concept_payload.definition
        MERGE (chunk)-[mention:MENTIONS_CONCEPT]->(concept)
          SET mention.confidence = concept_payload.confidence
        """,
        **hierarchy,
        **chunk,
        concepts=concepts,
    )
    await tx.run(
        """
        UNWIND $relationships AS rel_payload
        MATCH (source:Concept {name_norm: toLower(rel_payload.source)})
        MATCH (target:Concept {name_norm: toLower(rel_payload.target)})
        FOREACH (_ IN CASE WHEN rel_payload.relation = 'PREREQUISITE_OF' THEN [1] ELSE [] END |
          MERGE (source)-[rel:PREREQUISITE_OF]->(target)
            SET rel.relation = rel_payload.relation,
                rel.evidence = rel_payload.evidence,
                rel.confidence = rel_payload.confidence
        )
        FOREACH (_ IN CASE WHEN rel_payload.relation <> 'PREREQUISITE_OF' THEN [1] ELSE [] END |
          MERGE (source)-[rel:RELATED_TO]->(target)
            SET rel.relation = rel_payload.relation,
                rel.evidence = rel_payload.evidence,
                rel.confidence = rel_payload.confidence
        )
        """,
        relationships=relationships,
    )


__all__ = [
    "KnowledgeGraphClient",
    "KnowledgeGraphDisabledError",
    "close_neo4j",
    "get_neo4j_driver",
    "graph_client",
]
