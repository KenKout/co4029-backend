"""Knowledge graph value objects shared by builder + retrieval.

Public API surface for the AI primitive: dataclasses kept independent of
the SQLAlchemy ORM and Neo4j record shapes so callers can pass them
freely between ``ai/*`` and ``features/*``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    name: str
    type: str = "Concept"
    definition: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ConceptRelationship:
    source: str
    target: str
    relation: str = "RELATED_TO"
    evidence: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class KGSummary:
    concept_count: int
    relationship_count: int
    enabled: bool


@dataclass(frozen=True)
class KGContext:
    concepts: list[Concept] = field(default_factory=list)
    prerequisites: list[ConceptRelationship] = field(default_factory=list)
    related: list[ConceptRelationship] = field(default_factory=list)
    enabled: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.concepts and not self.prerequisites and not self.related


__all__ = [
    "Concept",
    "ConceptRelationship",
    "KGContext",
    "KGSummary",
]
