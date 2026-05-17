"""Courses-feature service layer (4 capability split per plan §4191).

* :mod:`.catalog`        — learner-side reads (published courses).
* :mod:`.authoring`      — teacher-side CRUD + reorder + soft-delete.
* :mod:`.assignment`     — HOD/Manager-side teacher assignment.
* :mod:`.administration` — IT Admin-side restore + processing audit.

Import-linter contract #1 forbids these modules from importing
``sqlalchemy`` at module level. Each sub-module declares
``AsyncSession`` under :class:`typing.TYPE_CHECKING` only and delegates
all DB access to the queries layer. INSERT/UPDATE flows construct ORM
instances via ``db.add(instance)`` / ``await db.flush()`` — these are
``AsyncSession`` methods, not module-level ``sqlalchemy`` imports.
"""

from __future__ import annotations

from abridgeai.features.courses.services import (
    administration,
    assignment,
    authoring,
    catalog,
)

__all__ = [
    "administration",
    "assignment",
    "authoring",
    "catalog",
]
