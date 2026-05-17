"""Knowledge graph retrieval for downstream generation pipelines.

Read-side counterpart to :mod:`abridgeai.ai.knowledge_graph.builder`. All
Cypher reads for KG-context lookups (lesson concepts, anchor traversal) live
here; ``infrastructure/neo4j`` is the slim driver layer.

When the knowledge graph is disabled or Neo4j is unreachable, the public
helpers return an empty :class:`KGContext` and never raise so generators
degrade gracefully to vector-only RAG.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from abridgeai.ai.knowledge_graph.schemas import (
    Concept,
    ConceptRelationship,
    KGContext,
)
from abridgeai.core.config import get_settings
from abridgeai.infrastructure.neo4j import (
    KnowledgeGraphClient,
    KnowledgeGraphDisabledError,
    graph_client,
)

logger = logging.getLogger(__name__)


MAX_CONCEPTS = 30
MAX_RELATIONSHIPS = 20


async def lesson_concepts(
    client: KnowledgeGraphClient,
    lesson_id: str | UUID,
) -> list[Concept]:
    async with client.session() as session:
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
            lesson_id=str(lesson_id),
        )
        records = [dict(record) async for record in result]
    return [
        Concept(
            name=record["name"],
            type=record.get("type") or "Concept",
            definition=record.get("definition"),
            confidence=record.get("confidence"),
        )
        for record in records
        if isinstance(record.get("name"), str)
    ]


async def lesson_concept_graph(
    client: KnowledgeGraphClient,
    lesson_id: str | UUID,
    *,
    depth: int = 2,
) -> tuple[list[Concept], list[ConceptRelationship]]:
    safe_depth = max(1, min(depth, 3))
    async with client.session() as session:
        result = await session.run(
            f"""
            MATCH (:Lesson {{id: $lesson_id}})-[:HAS_MATERIAL]->(:Material)
              -[:HAS_CHUNK]->(:Chunk)-[:MENTIONS_CONCEPT]->(seed:Concept)
            OPTIONAL MATCH path = (seed)-[:RELATED_TO|PREREQUISITE_OF*1..{safe_depth}]-(related:Concept)
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
            lesson_id=str(lesson_id),
        )
        record = await result.single()

    if record is None:
        return [], []
    raw_nodes = record["nodes"] or []
    raw_edges = record["edges"] or []

    concepts = [c for c in (_concept_from_record(n) for n in raw_nodes) if c is not None]
    relationships = [r for r in (_edge_from_record(e) for e in raw_edges) if r is not None]
    return concepts, relationships


async def retrieve_kg_context_for_anchors(
    anchor_concepts: list[str],
    *,
    depth: int = 2,
    client: KnowledgeGraphClient | None = None,
) -> KGContext:
    """Walk Neo4j outward from the named anchors and return the bounded context.

    Used by quiz / interview retrieval stages: given the lesson's top
    concepts, return up to :data:`MAX_CONCEPTS` connected concepts and up to
    :data:`MAX_RELATIONSHIPS` of each of (PREREQUISITE_OF, RELATED_TO).

    If the KG feature is disabled or Neo4j is unreachable, returns an empty
    ``KGContext`` (``enabled=False`` for the disabled case, ``enabled=True``
    with empty lists for transient failures so callers can distinguish).
    """
    names = [n.strip() for n in anchor_concepts if isinstance(n, str) and n.strip()]
    if not names:
        return KGContext(enabled=False)

    settings = get_settings()
    if not settings.knowledge_graph_enabled:
        return KGContext(enabled=False)

    safe_depth = max(1, min(depth, 3))

    if client is not None:
        return await _retrieve_with_client(client, names, safe_depth)

    try:
        async with graph_client() as owned_client:
            return await _retrieve_with_client(owned_client, names, safe_depth)
    except KnowledgeGraphDisabledError:
        return KGContext(enabled=False)
    except Exception as exc:  # pragma: no cover - graceful degradation when Neo4j is down
        logger.warning("Knowledge graph retrieval failed: %s", exc)
        return KGContext(enabled=True)


async def _retrieve_with_client(
    client: KnowledgeGraphClient,
    names: list[str],
    depth: int,
) -> KGContext:
    lowered = [n.lower() for n in names]
    async with client.session() as session:
        result = await session.run(
            f"""
            MATCH (anchor:Concept)
            WHERE anchor.name_norm IN $names
            OPTIONAL MATCH path = (anchor)-[:RELATED_TO|PREREQUISITE_OF*0..{depth}]-(related:Concept)
            WITH collect(DISTINCT anchor) + collect(DISTINCT related) AS node_list, collect(path) AS paths
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
            names=lowered,
        )
        record = await result.single()

    if record is None:
        return KGContext(enabled=True)

    raw_nodes = record["nodes"] or []
    raw_edges = record["edges"] or []

    concepts_by_id: dict[str, Concept] = {}
    for raw_node in raw_nodes:
        concept = _concept_from_record(raw_node)
        if concept is None:
            continue
        key = (raw_node.get("id") or concept.name).lower()
        concepts_by_id.setdefault(key, concept)

    prerequisites: dict[tuple[str, str], ConceptRelationship] = {}
    related: dict[tuple[str, str], ConceptRelationship] = {}
    for raw_edge in raw_edges:
        edge = _edge_from_record(raw_edge)
        if edge is None:
            continue
        bucket = prerequisites if edge.relation == "PREREQUISITE_OF" else related
        bucket.setdefault((edge.source, edge.target), edge)

    return KGContext(
        concepts=list(concepts_by_id.values())[:MAX_CONCEPTS],
        prerequisites=list(prerequisites.values())[:MAX_RELATIONSHIPS],
        related=list(related.values())[:MAX_RELATIONSHIPS],
        enabled=True,
    )


async def retrieve_kg_context_for_lessons(
    lesson_ids: Iterable[UUID | str],
    *,
    client: KnowledgeGraphClient | None = None,
) -> KGContext:
    """Aggregate ``lesson_concept_graph`` results across lessons.

    Mirrors the legacy ``retrieve_kg_context`` behaviour from
    ``backend/app/ai/haystack/components/kg_retrieval.py`` so the quiz /
    interview generation pipelines have a drop-in replacement.
    """
    ids = [str(lid) for lid in lesson_ids if lid is not None]
    if not ids:
        return KGContext(enabled=False)

    settings = get_settings()
    if not settings.knowledge_graph_enabled:
        return KGContext(enabled=False)

    if client is not None:
        return await _aggregate_lesson_graphs(client, ids)

    try:
        async with graph_client() as owned_client:
            return await _aggregate_lesson_graphs(owned_client, ids)
    except KnowledgeGraphDisabledError:
        return KGContext(enabled=False)
    except Exception as exc:  # pragma: no cover - graceful degradation when Neo4j is down
        logger.warning("Knowledge graph retrieval failed: %s", exc)
        return KGContext(enabled=True)


async def _aggregate_lesson_graphs(
    client: KnowledgeGraphClient,
    lesson_ids: list[str],
) -> KGContext:
    concepts_by_name: dict[str, Concept] = {}
    prerequisites: dict[tuple[str, str], ConceptRelationship] = {}
    related: dict[tuple[str, str], ConceptRelationship] = {}

    for lesson_id in lesson_ids:
        nodes, edges = await lesson_concept_graph(client, lesson_id)
        for concept in nodes:
            concepts_by_name.setdefault(concept.name.lower(), concept)
        for edge in edges:
            bucket = prerequisites if edge.relation == "PREREQUISITE_OF" else related
            bucket.setdefault((edge.source, edge.target), edge)

    return KGContext(
        concepts=list(concepts_by_name.values())[:MAX_CONCEPTS],
        prerequisites=list(prerequisites.values())[:MAX_RELATIONSHIPS],
        related=list(related.values())[:MAX_RELATIONSHIPS],
        enabled=True,
    )


def _concept_from_record(record: dict[str, Any]) -> Concept | None:
    label = record.get("label") or record.get("name")
    if not isinstance(label, str):
        return None
    return Concept(
        name=label,
        type=record.get("type") or "Concept",
        definition=record.get("definition"),
        confidence=_to_float(record.get("confidence")),
    )


def _edge_from_record(record: dict[str, Any]) -> ConceptRelationship | None:
    source = record.get("source")
    target = record.get("target")
    relation = record.get("relation") or "RELATED_TO"
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    if relation not in {"PREREQUISITE_OF", "RELATED_TO"}:
        relation = "RELATED_TO"
    return ConceptRelationship(
        source=source,
        target=target,
        relation=relation,
        evidence=record.get("evidence"),
        confidence=_to_float(record.get("confidence")),
    )


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "MAX_CONCEPTS",
    "MAX_RELATIONSHIPS",
    "lesson_concept_graph",
    "lesson_concepts",
    "retrieve_kg_context_for_anchors",
    "retrieve_kg_context_for_lessons",
]
