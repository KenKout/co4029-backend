"""Public API for the knowledge graph AI primitive.

External callers should import from ``abridgeai.ai.knowledge_graph``:

    from abridgeai.ai.knowledge_graph import (
        build_knowledge_graph_for_material_version,
        lesson_concepts,
        retrieve_kg_context_for_anchors,
        Concept,
        ConceptRelationship,
        KGSummary,
    )
"""

from abridgeai.ai.knowledge_graph.builder import (
    KG_BUILD_STAGE_NAME,
    KNOWLEDGE_GRAPH_SYSTEM_PROMPT,
    EnrichedChunk,
    HierarchyPayload,
    build_knowledge_graph_for_material_version,
    upsert_chunk_graph,
)
from abridgeai.ai.knowledge_graph.retrieval import (
    MAX_CONCEPTS,
    MAX_RELATIONSHIPS,
    lesson_concept_graph,
    lesson_concept_graph_preview,
    lesson_concepts,
    retrieve_kg_context_for_anchors,
    retrieve_kg_context_for_lessons,
)
from abridgeai.ai.knowledge_graph.schemas import (
    Concept,
    ConceptRelationship,
    KGContext,
    KGSummary,
)

__all__ = [
    "KG_BUILD_STAGE_NAME",
    "KNOWLEDGE_GRAPH_SYSTEM_PROMPT",
    "MAX_CONCEPTS",
    "MAX_RELATIONSHIPS",
    "Concept",
    "ConceptRelationship",
    "EnrichedChunk",
    "HierarchyPayload",
    "KGContext",
    "KGSummary",
    "build_knowledge_graph_for_material_version",
    "lesson_concept_graph",
    "lesson_concept_graph_preview",
    "lesson_concepts",
    "retrieve_kg_context_for_anchors",
    "retrieve_kg_context_for_lessons",
    "upsert_chunk_graph",
]
