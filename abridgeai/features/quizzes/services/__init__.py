"""Quizzes-feature service layer (3 capability split per plan §5964).

* :mod:`.authoring`  — teacher CRUD + revisions + ARQ-enqueue triggers.
* :mod:`.taking`     — student attempts (start / answer / submit / history).
* :mod:`.generation` — top-level pipeline dispatcher (called from ARQ worker).

Import-linter contract #1 forbids these modules from importing
``sqlalchemy`` at module level. Each sub-module declares
``AsyncSession`` under :class:`typing.TYPE_CHECKING` only and delegates
all DB access to the queries layer. INSERT/UPDATE flows construct ORM
instances via ``db.add(instance)`` / ``await db.flush()`` — these are
``AsyncSession`` methods, not module-level ``sqlalchemy`` imports.
"""

from __future__ import annotations

from abridgeai.features.quizzes.services import authoring, generation, taking

__all__ = ["authoring", "generation", "taking"]
