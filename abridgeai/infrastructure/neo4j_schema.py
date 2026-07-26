"""Idempotent Neo4j constraint + index DDL.

The graph has been running with **no constraints and no indexes**. Two
consequences, both live:

* Every ``MERGE (n:Label {id: ...})`` is an all-nodes label scan. That is
  invisible on a demo dataset and quadratic on a real one — the KG build does
  one MERGE per chunk per node type, so ingest cost grows with total graph
  size rather than document size.
* Nothing prevented duplicate ``Concept`` nodes. Without a uniqueness
  constraint, two concurrent ingests MERGE-ing the same concept name can both
  miss and both create, after which every later MERGE matches one of them
  arbitrarily and the mention edges split across twins.

The ``Concept`` constraint is on ``(org_id, name_norm)`` — the composite that
makes concepts tenant-local. Applying it also enforces, at the database
level, the isolation that ``builder.py`` implements in Cypher.

Called once from the FastAPI lifespan. Every statement is ``IF NOT EXISTS``,
so a restart is a no-op and a fresh database self-provisions.
"""

from __future__ import annotations

import logging

from abridgeai.infrastructure.neo4j import (
    KnowledgeGraphDisabledError,
    graph_client,
)

logger = logging.getLogger(__name__)

# Uniqueness doubles as an index in Neo4j, so a node keyed by one of these
# needs no separate index declaration.
_CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT course_id_unique IF NOT EXISTS "
    "FOR (n:Course) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT module_id_unique IF NOT EXISTS "
    "FOR (n:Module) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT lesson_id_unique IF NOT EXISTS "
    "FOR (n:Lesson) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT material_id_unique IF NOT EXISTS "
    "FOR (n:Material) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
    "FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    # The tenant-scoped concept identity. This is the constraint that makes
    # cross-org concept collision structurally impossible.
    "CREATE CONSTRAINT concept_org_name_unique IF NOT EXISTS "
    "FOR (n:Concept) REQUIRE (n.org_id, n.name_norm) IS UNIQUE",
)

_INDEXES: tuple[str, ...] = (
    # Anchor lookups filter on org before name; the composite constraint above
    # already covers that pair, but retrieval also scans concepts by org alone
    # when pruning orphans.
    "CREATE INDEX concept_org IF NOT EXISTS FOR (n:Concept) ON (n.org_id)",
    # ``prune_superseded_chunk_graph`` deletes by material version, and the
    # lesson-seeded retrieval filters chunks by org.
    "CREATE INDEX chunk_material_version IF NOT EXISTS "
    "FOR (n:Chunk) ON (n.material_version_id)",
    "CREATE INDEX chunk_org IF NOT EXISTS FOR (n:Chunk) ON (n.org_id)",
)


async def ensure_graph_schema() -> None:
    """Apply constraints + indexes. Never raises.

    Startup must not depend on Neo4j being reachable: the knowledge graph is
    an optional feature (``knowledge_graph_enabled``) and the API serves every
    non-KG route without it. A failure here is logged and the process
    continues — the DDL is retried on the next boot.
    """
    try:
        async with graph_client() as client, client.session() as session:
            for statement in (*_CONSTRAINTS, *_INDEXES):
                try:
                    await session.run(statement)
                except Exception:  # noqa: BLE001 -- one bad DDL must not skip the rest
                    logger.warning("neo4j schema statement failed: %s", statement, exc_info=True)
    except KnowledgeGraphDisabledError:
        logger.debug("knowledge graph disabled; skipping Neo4j schema DDL")
    except Exception:  # noqa: BLE001 -- must never crash startup
        logger.warning("could not apply Neo4j schema DDL", exc_info=True)
    else:
        logger.info("neo4j schema ensured (%d constraints, %d indexes)", len(_CONSTRAINTS), len(_INDEXES))


__all__ = ["ensure_graph_schema"]
