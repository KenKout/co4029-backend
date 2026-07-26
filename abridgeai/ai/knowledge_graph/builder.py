"""Knowledge graph builder.

Neo4j upsert + per-material orchestration of the KG build. Lives above
``infrastructure/neo4j`` (which now exposes only driver lifecycle) so all
Cypher writes for the ingestion path are centralised here. The LLM extraction
itself — prompts, response schema, payload normalisation — lives in
``extraction.py``.

Public entry point: :func:`build_knowledge_graph_for_material_version`. The
material/lesson/module/course rows must already be loaded so we can build
the hierarchy payload that anchors every chunk in Neo4j.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Protocol
from uuid import UUID

from neo4j import AsyncManagedTransaction
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.ai.knowledge_graph.consolidate import consolidate_material_concepts
from abridgeai.ai.knowledge_graph.extraction import (
    KG_BUILD_STAGE_NAME,
    KG_EXTRACTION_VERSION,
    KNOWLEDGE_GRAPH_SYSTEM_PROMPT,
    EnrichedChunk,
    _extract_concepts_from_chunk,
)
from abridgeai.ai.knowledge_graph.pruning import prune_superseded_chunk_graph
from abridgeai.ai.knowledge_graph.schemas import KGSummary
from abridgeai.ai.llm.gateway import LLMGateway
from abridgeai.core.config import get_settings
from abridgeai.infrastructure.neo4j import KnowledgeGraphClient

logger = logging.getLogger(__name__)


class _ConceptVocabulary:
    """Running list of concept names already extracted from this material.

    Fed back into every subsequent extraction prompt. Extraction is one
    isolated LLM call per chunk, so without this a concept introduced on slide
    11 and reused on slide 27 gets coined twice under two spellings and the two
    never link — which is how a 34-chunk deck lands as 20 disconnected
    components.

    Insertion-ordered so :meth:`recent_first` can hand the prompt the terms
    from the section currently being read, which are the ones the next chunk is
    most likely to reuse and so must survive the prompt's vocabulary cap.
    Deduped case-insensitively; the first spelling seen is the one offered back.
    """

    def __init__(self) -> None:
        self._names: list[str] = []
        self._seen: set[str] = set()

    def remember(self, name: str) -> None:
        cleaned = name.strip()
        key = cleaned.lower()
        if key and key not in self._seen:
            self._seen.add(key)
            self._names.append(cleaned)

    def remember_all(self, names: Iterable[str]) -> None:
        for name in names:
            self.remember(name)

    def recent_first(self) -> list[str]:
        return list(reversed(self._names))


async def _emit_progress(
    on_progress: Callable[[int, int], Awaitable[None]] | None,
    done: int,
    total: int,
) -> None:
    """Fire the caller's progress callback, swallowing its failures.

    Progress reporting is cosmetic; a caller whose callback raises must not
    take the whole KG build down with it.
    """
    if on_progress is None:
        return
    try:
        await on_progress(done, total)
    except Exception:  # noqa: BLE001 -- progress must never fail the build
        logger.debug("kg_build on_progress callback failed", exc_info=True)


class HierarchyPayload(Protocol):
    """Identifiers + titles for the Course→Module→Lesson→Material chain.

    Passed in by the caller (worker / ingestion service) so this module
    stays free of feature-layer ORM imports.
    """

    # Tenant scope. Every Concept node and every Concept lookup is keyed on
    # this: without it a ``Concept {name_norm}`` MERGE is GLOBAL, so
    # "Normalization" taught by org A and org B collapse into one node and a
    # RELATED_TO traversal walks straight out of the caller's tenant into
    # another customer's curriculum.
    organization_id: UUID
    course_id: UUID
    course_title: str
    module_id: UUID
    module_title: str
    lesson_id: UUID
    lesson_title: str
    material_id: UUID
    material_title: str
    material_type: str



def _hierarchy_dict(hierarchy: HierarchyPayload) -> dict[str, str]:
    return {
        "org_id": str(hierarchy.organization_id),
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


async def _already_built_previews_by_index(
    client: KnowledgeGraphClient,
    material_id: UUID,
) -> dict[int, str]:
    """Map of ``chunk_index -> text_preview`` already committed to Neo4j.

    ``upsert_chunk_graph`` MERGEs a ``Chunk`` node (linked under the
    material's stable ``Material`` node) for every chunk it finishes,
    stamping ``index`` + ``text_preview``. Querying them back lets the
    build **resume**: a run killed by ``job_timeout`` mid-way leaves its
    completed chunks committed in Neo4j (writes are per-chunk, outside the
    pipeline's Postgres transaction), so the next run can skip them instead
    of restarting from chunk 1. Without this, a document with more chunks
    than fit in one timeout window can never finish — it rebuilds from
    scratch every run and times out at the same place forever.

    Why index+preview, not chunk id: the pipeline deletes and recreates
    ``document_chunks`` on every run, so a chunk's Postgres UUID (and thus
    the Neo4j ``Chunk.id``) is NOT stable across runs — resuming by id
    would never match. ``chunk_index`` IS stable (deterministic chunking of
    the same source), and we verify the stored ``text_preview`` matches the
    current chunk's content before skipping, so a genuine reprocess with
    different content at the same index rebuilds rather than reusing stale
    concepts.

    Only nodes built by the CURRENT ``KG_EXTRACTION_VERSION`` are returned.
    Text-equality alone made this function a trap: after a prompt change the
    text is by definition unchanged, so every chunk matched, every chunk was
    skipped, and a deliberate rebuild returned the old prompt's concepts while
    reporting success. Filtering on the version means changing the prompt
    invalidates the graph on the next run with no manual Neo4j surgery — and
    nodes written before the property existed read as NULL, so they rebuild
    once and are correct thereafter.

    Keyed by the stable ``material_id`` (``Material`` nodes persist across
    versions). Best-effort: on any Neo4j error returns an empty map (full
    rebuild — correct, just slower), never raises.
    """
    try:
        async with client.session() as session:
            result = await session.run(
                """
                MATCH (m:Material {id: $material_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.index IS NOT NULL
                  AND c.kg_extraction_version = $extraction_version
                RETURN c.index AS idx, c.text_preview AS preview
                """,
                material_id=str(material_id),
                extraction_version=KG_EXTRACTION_VERSION,
            )
            return {
                int(record["idx"]): (record["preview"] or "")
                async for record in result
                if record["idx"] is not None
            }
    except Exception:  # noqa: BLE001 -- resume is an optimisation; degrade to full rebuild
        logger.warning("kg_build resume-scan failed; rebuilding all chunks", exc_info=True)
        return {}


async def _existing_concept_names(
    client: KnowledgeGraphClient,
    *,
    material_id: UUID,
    org_id: UUID,
) -> list[str]:
    """Concept names already attached to this material's chunks, newest first.

    Only used to reseed the prompt vocabulary on a resumed build. Ordered by
    descending chunk index so the terms nearest the resume point come first
    and survive ``_KNOWN_CONCEPT_LIMIT``.

    Best-effort like the resume scan itself: on any Neo4j error returns an
    empty list, which costs some cross-chunk linking on that one run but never
    fails the build.
    """
    try:
        async with client.session() as session:
            result = await session.run(
                """
                MATCH (m:Material {id: $material_id})-[:HAS_CHUNK]->(c:Chunk)
                      -[:MENTIONS_CONCEPT]->(k:Concept {org_id: $org_id})
                RETURN k.name AS name, max(c.index) AS last_index
                ORDER BY last_index DESC
                """,
                material_id=str(material_id),
                org_id=str(org_id),
            )
            return [
                str(record["name"]) async for record in result if record["name"]
            ]
    except Exception:  # noqa: BLE001 -- vocabulary reseed is an optimisation
        logger.warning("kg_build vocabulary reseed failed", exc_info=True)
        return []



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
          SET course.title = $course_title, course.org_id = $org_id
        MERGE (module:Module {id: $module_id})
          SET module.title = $module_title, module.org_id = $org_id
        MERGE (lesson:Lesson {id: $lesson_id})
          SET lesson.title = $lesson_title, lesson.org_id = $org_id
        MERGE (material:Material {id: $material_id})
          SET material.title = $material_title,
              material.type = $material_type,
              material.org_id = $org_id
        MERGE (chunk:Chunk {id: $chunk_id})
          SET chunk.index = $chunk_index,
              chunk.text_preview = $text_preview,
              chunk.org_id = $org_id,
              chunk.material_version_id = $material_version_id,
              // Stamped so the next run's resume scan can tell concepts built
              // by THIS prompt from ones built by an older one. Without it,
              // resume matches on unchanged text and a prompt change can never
              // take effect.
              chunk.kg_extraction_version = $kg_extraction_version
        MERGE (course)-[:CONTAINS_MODULE]->(module)
        MERGE (module)-[:CONTAINS_LESSON]->(lesson)
        MERGE (lesson)-[:HAS_MATERIAL]->(material)
        MERGE (material)-[:HAS_CHUNK]->(chunk)
        WITH chunk
        UNWIND $concepts AS concept_payload
        // Composite key: org_id FIRST. Keyed on name_norm alone this MERGE is
        // global, and every tenant's "normalization"/"index"/"transaction"
        // node is shared — which then makes RELATED_TO traversal a
        // cross-customer read. The uniqueness constraint in
        // ``ensure_graph_schema`` enforces the same pair.
        MERGE (concept:Concept {org_id: $org_id, name_norm: toLower(concept_payload.name)})
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
        // Both endpoints scoped to the same tenant, so an extracted edge can
        // never bridge two organizations' graphs.
        MATCH (source:Concept {org_id: $org_id, name_norm: toLower(rel_payload.source)})
        MATCH (target:Concept {org_id: $org_id, name_norm: toLower(rel_payload.target)})
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
        org_id=hierarchy["org_id"],
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
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> KGSummary:
    """Run KG extraction + upsert for every chunk of one material version.

    Each LLM call writes one ``ai_model_calls`` row tagged with
    ``stage_name='kg_build'`` and the supplied ``pipeline_run_id`` so the
    cost dashboard can roll up KG-build spend per pipeline run.

    ``on_progress`` is an optional async callback invoked after each chunk
    is processed with ``(done, total)``. The KG build is one sequential LLM
    call per chunk, so on a large document it can run for minutes at a
    single overall percent — the callback lets the caller surface live
    sub-progress ("42/85 chunks") so the UI doesn't look frozen. Callback
    errors are swallowed: progress reporting must never fail the build.

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

    vocabulary = _ConceptVocabulary()

    # Drop the previous version's chunk subgraph BEFORE the resume scan, so
    # the scan cannot match a stale preview and skip a chunk that now needs
    # rebuilding. Chunks of THIS version survive, keeping resume intact.
    await prune_superseded_chunk_graph(
        kg_client,
        material_id=hierarchy.material_id,
        material_version_id=material_version_id,
        org_id=hierarchy.organization_id,
    )

    # Resume support: skip chunks already committed to Neo4j by an earlier
    # run that was killed mid-build (e.g. job_timeout). Each skipped chunk
    # avoids an LLM call, so a re-run fast-forwards through completed work
    # and only spends its time budget on what's left — the build makes
    # forward progress across runs instead of restarting from chunk 1 and
    # timing out at the same place forever.
    # material_id (stable across versions) keys the Material node in Neo4j.
    material_id = hierarchy.material_id
    already_built = await _already_built_previews_by_index(kg_client, material_id)
    if already_built:
        logger.info(
            "kg_build resuming: %d chunk(s) already in Neo4j for material %s "
            "(%d total this run)",
            len(already_built),
            material_id,
            len(chunks),
        )
        # Seed the vocabulary from what the killed run already committed.
        # Without this a resumed build starts with an empty known-concept list
        # and re-coins variants of terms it established before the timeout —
        # the resume optimisation would otherwise cost graph connectivity.
        vocabulary.remember_all(
            await _existing_concept_names(
                kg_client,
                material_id=material_id,
                org_id=hierarchy.organization_id,
            )
        )

    total = len(chunks)
    resumed = 0
    for done, chunk in enumerate(chunks, start=1):
        if chunk.material_version_id != material_version_id:
            logger.warning(
                "skipping chunk %s: material_version_id mismatch (expected %s, got %s)",
                chunk.id,
                material_version_id,
                chunk.material_version_id,
            )
            continue

        # Resume: skip a chunk only if a node with the SAME index AND the same
        # text_preview (content[:300]) already exists — matching both guards
        # against reusing stale concepts if a reprocess changed the content at
        # this index. Still fire the progress callback so the UI count advances
        # through the resumed range instead of appearing stalled.
        prior_preview = already_built.get(chunk.chunk_index)
        if prior_preview is not None and prior_preview == chunk.content[:300]:
            resumed += 1
            await _emit_progress(on_progress, done, total)
            continue

        concepts, relationships = await _extract_concepts_from_chunk(
            chunk,
            db=db,
            llm_gateway=llm_gateway,
            pipeline_run_id=pipeline_run_id,
            parent_job_id=parent_job_id,
            known_concepts=vocabulary.recent_first(),
        )
        await upsert_chunk_graph(
            kg_client,
            hierarchy=hierarchy_payload,
            chunk={
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "text_preview": chunk.content[:300],
                # Stamped so ``prune_material_version_graph`` can find this
                # node again after ``_persist_chunks`` has thrown its Postgres
                # row away and minted a fresh id on re-ingest.
                "material_version_id": str(material_version_id),
                "kg_extraction_version": KG_EXTRACTION_VERSION,
            },
            concepts=concepts,
            relationships=relationships,
        )
        concept_names.update(concept["name"].lower() for concept in concepts)
        vocabulary.remember_all(concept["name"] for concept in concepts)
        relationship_keys.update(
            (rel["source"].lower(), rel["target"].lower(), rel["relation"]) for rel in relationships
        )

        await _emit_progress(on_progress, done, total)

    # Fold duplicate spellings into one node now that every chunk has been
    # seen. This runs last on purpose: mid-build it would merge against a
    # partial graph and pick a canonical node by mention counts that are still
    # climbing, and a resumed run would merge the same material twice under
    # different winners.
    merged = await consolidate_material_concepts(
        kg_client,
        material_id=material_id,
        org_id=hierarchy.organization_id,
    )

    return KGSummary(
        # ``concept_names`` counts distinct spellings *extracted*; consolidation
        # then removes the ones that were the same concept twice. Report what
        # the graph actually holds, not what the LLM emitted.
        concept_count=max(len(concept_names) - merged, 0),
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
