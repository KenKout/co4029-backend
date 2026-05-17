"""Knowledge graph builder.

Per-chunk LLM extraction + Neo4j upsert. Lives above ``infrastructure/neo4j``
(which now exposes only driver lifecycle) so all Cypher writes for the
ingestion path are centralised here.

Public entry point: :func:`build_knowledge_graph_for_material_version`. The
material/lesson/module/course rows must already be loaded so we can build
the hierarchy payload that anchors every chunk in Neo4j.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol
from uuid import UUID

from neo4j import AsyncManagedTransaction
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.knowledge_graph.schemas import KGSummary
from abridgeai.ai.llm.gateway import LLMGateway
from abridgeai.ai.llm.roles import LLMRole
from abridgeai.core.config import get_settings
from abridgeai.infrastructure.neo4j import KnowledgeGraphClient

logger = logging.getLogger(__name__)


KG_BUILD_STAGE_NAME = "kg_build"

KNOWLEDGE_GRAPH_SYSTEM_PROMPT = (
    "You extract concise course knowledge graphs from LMS source chunks.\n"
    "Return one JSON object with entities and relationships arrays.\n"
    "Use only concepts supported by the source text.\n"
    "Prefer durable learning concepts over generic words.\n"
    "Relationships must use RELATED_TO or PREREQUISITE_OF.\n"
)


class EnrichedChunk(Protocol):
    """Forward-compatible structural type for a chunk awaiting KG extraction.

    T2.8 will land the concrete ``EnrichedChunk`` dataclass under
    ``ai/chunking``; until then we accept anything that quacks with the four
    fields the KG builder reads. The legacy ORM ``DocumentChunk`` already
    matches.
    """

    id: UUID
    chunk_index: int
    content: str
    material_version_id: UUID


class HierarchyPayload(Protocol):
    """Identifiers + titles for the Course→Module→Lesson→Material chain.

    Passed in by the caller (worker / ingestion service) so this module
    stays free of feature-layer ORM imports.
    """

    course_id: UUID
    course_title: str
    module_id: UUID
    module_title: str
    lesson_id: UUID
    lesson_title: str
    material_id: UUID
    material_title: str
    material_type: str


class _GeneratedConcept(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: str = Field(default="Concept", max_length=60)
    definition: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("name", "type", "definition", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class _GeneratedRelationship(BaseModel):
    source: str = Field(min_length=1, max_length=120)
    target: str = Field(min_length=1, max_length=120)
    relation: Literal["RELATED_TO", "PREREQUISITE_OF"] = "RELATED_TO"
    evidence: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("source", "target", "evidence", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class _GeneratedKnowledgeGraph(BaseModel):
    entities: list[_GeneratedConcept] = Field(default_factory=list)
    relationships: list[_GeneratedRelationship] = Field(default_factory=list)


def _normalize_kg_payload(
    payload: dict[str, Any] | list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return [], []
    try:
        generated = _GeneratedKnowledgeGraph.model_validate(payload)
    except (TypeError, ValidationError):
        return [], []

    concept_names = {concept.name.lower() for concept in generated.entities}
    relationships = [
        rel.model_dump()
        for rel in generated.relationships
        if rel.source.lower() in concept_names and rel.target.lower() in concept_names
    ]
    return [concept.model_dump() for concept in generated.entities], relationships


def _build_user_prompt(chunk: EnrichedChunk) -> str:
    excerpt = chunk.content[:1800]
    return f"""Source chunks:
[{chunk.id}] {excerpt}

Return JSON in this shape:
{{
  "entities": [
    {{
      "name": "Binary Search",
      "type": "Concept",
      "definition": "A search algorithm for sorted collections.",
      "confidence": 0.92
    }}
  ],
  "relationships": [
    {{
      "source": "Sorted Array",
      "target": "Binary Search",
      "relation": "PREREQUISITE_OF",
      "evidence": "Binary search requires sorted input.",
      "confidence": 0.88
    }}
  ]
}}
"""


def _hierarchy_dict(hierarchy: HierarchyPayload) -> dict[str, str]:
    return {
        "course_id": str(hierarchy.course_id),
        "course_title": hierarchy.course_title,
        "module_id": str(hierarchy.module_id),
        "module_title": hierarchy.module_title,
        "lesson_id": str(hierarchy.lesson_id),
        "lesson_title": hierarchy.lesson_title,
        "material_id": str(hierarchy.material_id),
        "material_title": hierarchy.material_title,
        "material_type": hierarchy.material_type,
    }


async def _extract_concepts_from_chunk(
    chunk: EnrichedChunk,
    *,
    db: AsyncSession,
    llm_gateway: LLMGateway,
    pipeline_run_id: UUID | None,
    parent_job_id: UUID | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = await llm_gateway.generate_json(
        role=LLMRole.KG_EXTRACTION,
        system_prompt=KNOWLEDGE_GRAPH_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(chunk),
        db=db,
        stage_name=KG_BUILD_STAGE_NAME,
        pipeline_run_id=pipeline_run_id,
        parent_job_id=parent_job_id,
    )
    return _normalize_kg_payload(result.content_json)


async def upsert_chunk_graph(
    client: KnowledgeGraphClient,
    *,
    hierarchy: dict[str, Any],
    chunk: dict[str, Any],
    concepts: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    async with client.session() as session:
        await session.execute_write(
            _upsert_chunk_graph_tx,
            hierarchy,
            chunk,
            concepts,
            relationships,
        )


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


async def build_knowledge_graph_for_material_version(
    material_version_id: UUID,
    chunks: list[EnrichedChunk],
    *,
    hierarchy: HierarchyPayload,
    pipeline_run_id: UUID | None,
    db: AsyncSession,
    kg_client: KnowledgeGraphClient,
    llm_gateway: LLMGateway,
    parent_job_id: UUID | None = None,
) -> KGSummary:
    """Run KG extraction + upsert for every chunk of one material version.

    Each LLM call writes one ``ai_model_calls`` row tagged with
    ``stage_name='kg_build'`` and the supplied ``pipeline_run_id`` so the
    cost dashboard can roll up KG-build spend per pipeline run.

    Returns a summary; never raises ``KnowledgeGraphDisabledError`` — when
    the feature flag is off we short-circuit to ``KGSummary(enabled=False)``.
    """
    settings = get_settings()
    if not settings.knowledge_graph_enabled:
        return KGSummary(concept_count=0, relationship_count=0, enabled=False)

    if not chunks:
        return KGSummary(concept_count=0, relationship_count=0, enabled=True)

    hierarchy_payload = _hierarchy_dict(hierarchy)
    concept_names: set[str] = set()
    relationship_keys: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        if chunk.material_version_id != material_version_id:
            logger.warning(
                "skipping chunk %s: material_version_id mismatch (expected %s, got %s)",
                chunk.id,
                material_version_id,
                chunk.material_version_id,
            )
            continue

        concepts, relationships = await _extract_concepts_from_chunk(
            chunk,
            db=db,
            llm_gateway=llm_gateway,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
        )
        await upsert_chunk_graph(
            kg_client,
            hierarchy=hierarchy_payload,
            chunk={
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "text_preview": chunk.content[:300],
            },
            concepts=concepts,
            relationships=relationships,
        )
        concept_names.update(concept["name"].lower() for concept in concepts)
        relationship_keys.update(
            (rel["source"].lower(), rel["target"].lower(), rel["relation"]) for rel in relationships
        )

    return KGSummary(
        concept_count=len(concept_names),
        relationship_count=len(relationship_keys),
        enabled=True,
    )


__all__ = [
    "EnrichedChunk",
    "HierarchyPayload",
    "KG_BUILD_STAGE_NAME",
    "KNOWLEDGE_GRAPH_SYSTEM_PROMPT",
    "build_knowledge_graph_for_material_version",
    "upsert_chunk_graph",
]
