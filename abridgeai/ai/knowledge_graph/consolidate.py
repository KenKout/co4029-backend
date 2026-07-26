"""Post-extraction concept consolidation.

Extraction is one isolated LLM call per chunk and the Neo4j upsert keys a
``Concept`` on ``toLower(name)``, so the graph accumulates one node per
*spelling*. On a single 34-slide lecture that produced, among others:

    Expert system            / Expert systems
    Artificial neural networks / Artificial neural networks (ANN)
    Executive Information Systems / Executive Information System (EIS)
    Supply chain management system / Supply chain management (SCM) systems
    GSS                      / Group support systems (GSS)
    Simon's decision-making process / Simon’s decision-making process

Each pair is one concept wearing two names, and the damage is not cosmetic:
the extracted relationship lands on whichever spelling that chunk happened to
use, so one node ends up connected and its twin sits at degree 0. Roughly half
the orphans in a freshly built graph are this, not genuine extraction misses.

This pass merges them. It is deliberately **deterministic** — a normalisation
key, not an embedding threshold. Concept merging is destructive and hard to
audit after the fact; a rule that can be read off the name is one a teacher can
be shown ("these two are the same because they normalise identically"), and it
cannot quietly collapse two genuinely different ideas the way a cosine gate
tuned on one corpus will on the next.

What it deliberately does NOT do
--------------------------------
Strip generic head nouns. "Decision support" and "Decision support systems"
normalise differently and stay separate, which also leaves
"Enterprise Resource Management" apart from "Enterprise resource management
(ERM) systems". That is the conservative direction: an unmerged pair costs one
edge, a wrongly merged pair silently rewrites what the course says.

Acronyms are resolved only when unambiguous *within the tenant*. This same
lecture contains both "Executive Information System (EIS)" and "Enterprise
information systems (EIS)" — a naive acronym map would fuse two distinct
concepts, so an acronym claimed by more than one expansion is dropped from the
index entirely.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any
from uuid import UUID

from neo4j import AsyncManagedTransaction

from abridgeai.infrastructure.neo4j import KnowledgeGraphClient

logger = logging.getLogger(__name__)

_PAREN_RE = re.compile(r"\(([^)]*)\)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# An acronym worth indexing: 2-6 letters, as written on the slide (upper case).
# One letter is noise; past six it is not an acronym, it is a word.
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")


def _singularize(token: str) -> str:
    """Crude English singulariser, tuned for course-material noun phrases.

    Only has to be *consistent*, not linguistically right: it feeds a grouping
    key, so mapping "analyses" to something odd is harmless as long as every
    spelling of it maps there too. The guards that matter are the ones that
    leave already-singular words alone — "process", "analysis", "business",
    "status" all end in ``s`` and must not lose it.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith(("ss", "is", "us")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def canonical_key(name: str) -> str:
    """Normalisation key: two names sharing one are the same concept.

    Drops parentheticals (so ``Group support systems (GSS)`` keys as
    ``group support system``), folds accents and curly apostrophes, reduces
    punctuation to spaces, and singularises each token.
    """
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    without_parens = _PAREN_RE.sub(" ", folded)
    lowered = _NON_ALNUM_RE.sub(" ", without_parens.lower())
    tokens = [_singularize(t) for t in lowered.split() if t]
    return " ".join(tokens)


def _acronyms_in(name: str) -> list[str]:
    """Upper-case acronyms written parenthetically in ``name``."""
    out: list[str] = []
    for inner in _PAREN_RE.findall(name):
        candidate = inner.strip()
        if _ACRONYM_RE.match(candidate):
            out.append(candidate.lower())
    return out


def plan_merges(concepts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group ``concepts`` into ``canonical_key -> members`` (groups of 2+ only).

    ``concepts`` items carry ``name``, ``mentions``, ``degree`` and
    ``has_definition``. Pure function so the grouping rules are testable
    without a graph.
    """
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for concept in concepts:
        by_key[canonical_key(str(concept["name"]))].append(concept)

    # Acronym index, built only from unambiguous claims. ``expansions`` tracks
    # every distinct key an acronym is written against; anything claimed twice
    # is poisoned and never resolves.
    expansions: dict[str, set[str]] = defaultdict(set)
    for concept in concepts:
        key = canonical_key(str(concept["name"]))
        for acronym in _acronyms_in(str(concept["name"])):
            expansions[acronym].add(key)
    acronym_to_key = {
        acronym: next(iter(keys)) for acronym, keys in expansions.items() if len(keys) == 1
    }

    # Fold bare-acronym nodes ("GSS") into their expansion's group.
    for acronym, target_key in acronym_to_key.items():
        members = by_key.get(acronym)
        if not members or acronym == target_key:
            continue
        by_key[target_key].extend(members)
        del by_key[acronym]

    return {key: members for key, members in by_key.items() if len(members) > 1}


def choose_canonical(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the surviving node: the spelling the document actually leans on.

    Ordered by mention count, then existing degree, then whether it carries a
    definition, then name length. Mentions lead because that is the spelling a
    reader of this course meets most often; degree breaks ties toward the node
    that needs the fewest edges rewired.
    """
    return max(
        members,
        key=lambda c: (
            int(c.get("mentions") or 0),
            int(c.get("degree") or 0),
            1 if c.get("has_definition") else 0,
            len(str(c["name"])),
        ),
    )


async def consolidate_material_concepts(
    client: KnowledgeGraphClient,
    *,
    material_id: UUID,
    org_id: UUID,
) -> int:
    """Merge duplicate-spelling concepts reachable from one material's chunks.

    Returns the number of nodes merged away. Best-effort: on any Neo4j error
    the graph is left exactly as extraction wrote it and the ingest continues,
    because a fragmented graph is a degraded feature while a failed ingest is
    a lost upload.

    Scoped by ``org_id`` on every read and write — the tenant boundary the
    ``Concept`` composite key exists to enforce.
    """
    try:
        async with client.session() as session:
            result = await session.run(
                """
                MATCH (m:Material {id: $material_id})-[:HAS_CHUNK]->(:Chunk)
                      -[:MENTIONS_CONCEPT]->(k:Concept {org_id: $org_id})
                WITH DISTINCT k
                OPTIONAL MATCH (k)<-[mention:MENTIONS_CONCEPT]-()
                WITH k, count(mention) AS mentions
                OPTIONAL MATCH (k)-[rel:RELATED_TO|PREREQUISITE_OF]-()
                RETURN k.name AS name,
                       mentions AS mentions,
                       count(rel) AS degree,
                       k.definition IS NOT NULL AS has_definition
                """,
                material_id=str(material_id),
                org_id=str(org_id),
            )
            concepts = [dict(record) async for record in result]

            groups = plan_merges(concepts)
            if not groups:
                return 0

            merged = 0
            for key, members in groups.items():
                canonical = choose_canonical(members)
                duplicates = [
                    str(m["name"]) for m in members if m["name"] != canonical["name"]
                ]
                if not duplicates:
                    continue
                await session.execute_write(
                    _merge_concepts_tx,
                    org_id=str(org_id),
                    canonical_name=str(canonical["name"]),
                    duplicate_names=duplicates,
                )
                merged += len(duplicates)
                logger.debug(
                    "kg consolidate: %s <- %s (key=%r)",
                    canonical["name"],
                    duplicates,
                    key,
                )
    except Exception:  # noqa: BLE001 -- consolidation is hygiene; never fail an ingest
        logger.warning(
            "kg consolidation failed for material %s; duplicate concepts retained",
            material_id,
            exc_info=True,
        )
        return 0

    if merged:
        logger.info(
            "kg consolidate: merged %d duplicate concept node(s) for material %s",
            merged,
            material_id,
        )
    return merged


async def _merge_concepts_tx(
    tx: AsyncManagedTransaction,
    *,
    org_id: str,
    canonical_name: str,
    duplicate_names: list[str],
) -> None:
    """Rewire every duplicate's edges onto the canonical node, then delete it.

    Order matters. Mentions and relationships are re-MERGEd onto the canonical
    node *before* the duplicate is deleted, so a failure mid-transaction rolls
    back to the pre-merge graph rather than leaving a concept with no way back
    to its chunks. ``DETACH DELETE`` then removes the now-redundant duplicate
    along with the original copies of the edges we just re-pointed.

    Self-loops are filtered explicitly: an edge that already ran between the
    duplicate and the canonical node would otherwise land as
    ``canonical -> canonical`` and show up in the UI as a concept that is its
    own prerequisite.
    """
    await tx.run(
        """
        MATCH (canon:Concept {org_id: $org_id, name_norm: toLower($canonical_name)})
        UNWIND $duplicate_names AS dup_name
        MATCH (dup:Concept {org_id: $org_id, name_norm: toLower(dup_name)})
        WHERE dup <> canon

        // 1. Mentions: every chunk that pointed at the duplicate now points at
        //    the canonical node. MERGE so a chunk mentioning both spellings
        //    ends with one edge, not two.
        CALL (dup, canon) {
          MATCH (c:Chunk)-[m:MENTIONS_CONCEPT]->(dup)
          MERGE (c)-[kept:MENTIONS_CONCEPT]->(canon)
            SET kept.confidence = coalesce(kept.confidence, m.confidence)
        }

        // 2. Outgoing concept edges.
        CALL (dup, canon) {
          MATCH (dup)-[r:RELATED_TO|PREREQUISITE_OF]->(other:Concept)
          WHERE other <> canon
          FOREACH (_ IN CASE WHEN type(r) = 'PREREQUISITE_OF' THEN [1] ELSE [] END |
            MERGE (canon)-[kept:PREREQUISITE_OF]->(other)
              SET kept.relation = 'PREREQUISITE_OF',
                  kept.evidence = coalesce(kept.evidence, r.evidence),
                  kept.confidence = coalesce(kept.confidence, r.confidence)
          )
          FOREACH (_ IN CASE WHEN type(r) = 'RELATED_TO' THEN [1] ELSE [] END |
            MERGE (canon)-[kept:RELATED_TO]->(other)
              SET kept.relation = 'RELATED_TO',
                  kept.evidence = coalesce(kept.evidence, r.evidence),
                  kept.confidence = coalesce(kept.confidence, r.confidence)
          )
        }

        // 3. Incoming concept edges.
        CALL (dup, canon) {
          MATCH (other:Concept)-[r:RELATED_TO|PREREQUISITE_OF]->(dup)
          WHERE other <> canon
          FOREACH (_ IN CASE WHEN type(r) = 'PREREQUISITE_OF' THEN [1] ELSE [] END |
            MERGE (other)-[kept:PREREQUISITE_OF]->(canon)
              SET kept.relation = 'PREREQUISITE_OF',
                  kept.evidence = coalesce(kept.evidence, r.evidence),
                  kept.confidence = coalesce(kept.confidence, r.confidence)
          )
          FOREACH (_ IN CASE WHEN type(r) = 'RELATED_TO' THEN [1] ELSE [] END |
            MERGE (other)-[kept:RELATED_TO]->(canon)
              SET kept.relation = 'RELATED_TO',
                  kept.evidence = coalesce(kept.evidence, r.evidence),
                  kept.confidence = coalesce(kept.confidence, r.confidence)
          )
        }

        // 4. Keep the merged spelling searchable, and inherit a definition if
        //    the canonical node never got one.
        SET canon.definition = coalesce(canon.definition, dup.definition),
            canon.aliases =
              [a IN coalesce(canon.aliases, []) WHERE a <> dup.name] + [dup.name]

        DETACH DELETE dup
        """,
        org_id=org_id,
        canonical_name=canonical_name,
        duplicate_names=duplicate_names,
    )


__all__ = [
    "canonical_key",
    "choose_canonical",
    "consolidate_material_concepts",
    "plan_merges",
]
