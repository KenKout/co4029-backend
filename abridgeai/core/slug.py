"""Shared slug helpers: Vietnamese-aware slugify + uniqueness suffixing.

Used by course authoring (lessons today) and, since migration 0086, by quiz
and interview-config items so every student-facing URL segment is generated
the same way: ``Operating System`` → ``operating-system``, ``Quiz 1.0`` →
``quiz-10``, and collisions get ``-1``, ``-2``, … appended (incrementing
from 1, never random).
"""

from __future__ import annotations

import re
import unicodedata

_MAX_SLUG_LEN = 100


def slugify(value: str) -> str:
    """ASCII, lowercase, hyphen-joined — Vietnamese letters folded to ASCII.

    Mirrors ``features/courses/ingest/syllabus._slugify`` (kept there for
    import-shape reasons); this is the canonical copy for runtime entity
    slugs.
    """
    folded = unicodedata.normalize("NFD", value)
    ascii_only = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    ascii_only = ascii_only.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")[:_MAX_SLUG_LEN]


def unique_slug(base: str, taken: set[str]) -> str:
    """Return ``base``, or ``base-1``, ``base-2``, … for the first free slot.

    Collision policy (product decision): append an integer incrementing from
    1 — deterministic and readable, never a random fragment. ``taken`` must
    contain the slugs already used within the uniqueness scope (a module for
    curriculum items, an org for courses).
    """
    candidate = base or "item"
    if candidate not in taken:
        return candidate
    n = 1
    while f"{candidate}-{n}" in taken:
        n += 1
    return f"{candidate}-{n}"


__all__ = ["slugify", "unique_slug"]
