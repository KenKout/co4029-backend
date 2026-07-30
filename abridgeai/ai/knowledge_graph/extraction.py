"""Per-chunk LLM extraction for the knowledge graph.

Split out of ``builder.py`` — which keeps the Neo4j write path and the
per-material orchestration — so both stay under the 500-LOC module budget
enforced by ``test_no_god_files_under_abridgeai_ai``.

This module owns everything between "here is one chunk" and "here are its
concepts and relationships": the prompts, the response schema, and the
normalisation that turns a model's JSON into upsertable payloads. It performs
no graph writes.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.llm.gateway import LLMGateway
from abridgeai.ai.llm.roles import LLMRole

KG_BUILD_STAGE_NAME = "kg_build"

# Bump whenever the extraction prompt, the response schema, or the
# normalisation below changes in a way that would produce different concepts
# from the same text.
#
# This is what makes the build's resume safe. Resume skips a chunk whose text
# is unchanged, which is right for continuing a run killed by ``job_timeout``
# and WRONG after a prompt change: the text is identical, so every chunk is
# skipped and a "rebuild" silently returns the concepts the old prompt
# produced. The version is stamped on each ``Chunk`` node and compared on
# resume, so changing the prompt invalidates the graph automatically instead
# of requiring someone to remember to clear Neo4j by hand.
KG_EXTRACTION_VERSION = "kg-v2"

KNOWLEDGE_GRAPH_SYSTEM_PROMPT = (
    "You extract concise course knowledge graphs from LMS source chunks.\n"
    "Return one JSON object with entities and relationships arrays.\n"
    "Use only concepts supported by the source text.\n"
    "Prefer durable learning concepts over generic words.\n"
    "Relationships must use RELATED_TO or PREREQUISITE_OF.\n"
    "\n"
    "Every entity you list MUST appear in at least one relationship. If the "
    "source text does not support any relationship for a term, leave the term "
    "out entirely — an unconnected entity is worse than a missing one.\n"
    "Do not extract the slide's running header or section title as an entity "
    "(e.g. 'Managers and computerized support systems'); it names the page, "
    "not a concept. Extract what the page teaches.\n"
    "You are given the concepts already extracted from earlier chunks of this "
    "same document. When this chunk discusses one of them, reuse that exact "
    "spelling instead of coining a variant, and relate new concepts to it "
    "where the text supports the link — that is how the per-chunk extractions "
    "become one graph rather than many disconnected islands.\n"
)

# Cap on the known-concept vocabulary injected into each prompt. Large enough
# to carry a lecture's worth of recurring terms, small enough that the list
# cannot crowd out the chunk itself in the context window. Most-recent-first
# so a long document keeps the terms still in play rather than the title page's.
_KNOWN_CONCEPT_LIMIT = 60

class EnrichedChunk(Protocol):
    """Forward-compatible structural type for a chunk awaiting KG extraction.

    T2.8 will land the concrete ``EnrichedChunk`` dataclass under
    ``ai/chunking``; until then we accept anything that quacks with the four
    fields the KG builder reads. The legacy ORM ``DocumentChunk`` already
    matches.

    The chunking-enrichment payload (``section_title`` / ``context_sentence``
    / ``key_concepts``) is read opportunistically by :func:`_chunk_semantic`
    off whichever attribute carries it, and is deliberately NOT part of this
    contract: extraction degrades to the bare chunk text when it is absent
    rather than failing.
    """

    id: UUID
    chunk_index: int
    content: str
    material_version_id: UUID

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

    concepts = [concept.model_dump() for concept in generated.entities]
    concept_names = {concept.name.lower() for concept in generated.entities}

    # An endpoint named in a relationship but missing from ``entities`` used to
    # cost us the whole edge. That is the wrong trade: the model asserting
    # "Expert systems enhance MSS technologies" IS an assertion that
    # "MSS technologies" is a concept — it just forgot to list it twice. The
    # upsert MATCHes both endpoints by name, so a dropped entity silently
    # dropped the edge too, and both concepts lost a connection they had
    # earned. Promote the missing endpoint to an entity instead; it carries no
    # definition, which the consolidation pass then folds into whichever
    # spelling of the same concept does.
    relationships: list[dict[str, Any]] = []
    for rel in generated.relationships:
        # A concept is not its own prerequisite, and "X is related to X" is a
        # fact about nothing. The upsert MERGEs by name, so a self-loop becomes
        # a visible edge from a node to itself in the lesson graph a teacher
        # reviews. Drop it here rather than filtering at render time.
        if rel.source.lower() == rel.target.lower():
            continue
        for endpoint in (rel.source, rel.target):
            if endpoint.lower() not in concept_names:
                concept_names.add(endpoint.lower())
                concepts.append(
                    {
                        "name": endpoint,
                        "type": "Concept",
                        "definition": None,
                        "confidence": rel.confidence,
                    }
                )
        relationships.append(rel.model_dump())

    return concepts, relationships


def _chunk_semantic(chunk: EnrichedChunk) -> dict[str, Any]:
    """Best-effort read of the chunking-enrichment payload for ``chunk``.

    Two shapes reach this module. The ORM ``DocumentChunk`` keeps the JSONB
    blob on ``metadata_json`` (``metadata`` is reserved by SQLAlchemy's
    ``DeclarativeBase``) with the enrichment nested under a ``semantic`` key;
    the ``ai/chunking`` ``EnrichedChunk`` dataclass exposes the same payload
    directly as ``semantic_metadata``. Neither is guaranteed — a chunk whose
    enrichment call failed falls back to the rule-based promotion — so every
    read here is defensive and an empty dict is a normal outcome.
    """
    direct = getattr(chunk, "semantic_metadata", None)
    if isinstance(direct, dict) and direct:
        return direct
    md = getattr(chunk, "metadata_json", None)
    if not isinstance(md, dict):
        md = getattr(chunk, "metadata", None)
    if isinstance(md, dict):
        semantic = md.get("semantic")
        if isinstance(semantic, dict):
            return semantic
    return {}


def _build_user_prompt(
    chunk: EnrichedChunk,
    known_concepts: list[str] | None = None,
) -> str:
    """Render the extraction prompt for one chunk.

    The chunk text alone is not enough context to extract a *graph* from. A
    lecture slide is written to be read in sequence: "Intelligence / Design /
    Choice" means nothing without knowing the page sits under Simon's
    decision-making process. Passing the enrichment stage's ``section_title``
    and ``context_sentence`` — already computed upstream, already paid for —
    tells the model what the page is about, and ``known_concepts`` tells it
    what the rest of the document already calls things, so it links to the
    existing node instead of minting a near-duplicate.
    """
    excerpt = chunk.content[:1800]
    semantic = _chunk_semantic(chunk)

    context_lines = []
    section_title = semantic.get("section_title")
    if isinstance(section_title, str) and section_title.strip():
        context_lines.append(f"Section: {section_title.strip()}")
    context_sentence = semantic.get("context_sentence")
    if isinstance(context_sentence, str) and context_sentence.strip():
        context_lines.append(f"Context: {context_sentence.strip()}")
    key_concepts = semantic.get("key_concepts")
    if isinstance(key_concepts, list):
        terms = [str(t).strip() for t in key_concepts if str(t).strip()]
        if terms:
            context_lines.append(f"Key terms on this page: {', '.join(terms)}")
    context_block = ("\n".join(context_lines) + "\n\n") if context_lines else ""

    known_block = ""
    if known_concepts:
        listed = ", ".join(known_concepts[:_KNOWN_CONCEPT_LIMIT])
        known_block = (
            f"Concepts already extracted from earlier chunks of this document "
            f"(reuse these exact spellings when this chunk discusses them):\n"
            f"{listed}\n\n"
        )

    return f"""{context_block}{known_block}Source chunks:
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

async def _extract_concepts_from_chunk(
    chunk: EnrichedChunk,
    *,
    db: AsyncSession,
    llm_gateway: LLMGateway,
    pipeline_run_id: UUID | None,
    parent_job_id: UUID | None,
    known_concepts: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = await llm_gateway.generate_json(
        role=LLMRole.KG_EXTRACTION,
        system_prompt=KNOWLEDGE_GRAPH_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(chunk, known_concepts),
        db=db,
        stage_name=KG_BUILD_STAGE_NAME,
        pipeline_run_id=pipeline_run_id,
        parent_job_id=parent_job_id,
    )
    return _normalize_kg_payload(result.content_json)

__all__ = [
    "KG_BUILD_STAGE_NAME",
    "KNOWLEDGE_GRAPH_SYSTEM_PROMPT",
    "EnrichedChunk",
]
