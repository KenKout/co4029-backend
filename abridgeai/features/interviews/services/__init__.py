"""Interviews-feature service layer (T6.11).

* :mod:`.authoring`  — teacher CRUD + manual question/outcome edits +
  ARQ-enqueue triggers (``start_generation_run``), re-exporting the
  sibling modules it was split into for the LOC ratchet: :mod:`.bank`
  (course question bank), :mod:`.outcomes`, :mod:`.generation_runs`,
  and the shared guards in :mod:`._shared`.
* :mod:`.taking`     — student session lifecycle (``start_session``,
  ``take_session_step``, ``submit_session``) with the runtime
  follow-up hook + post-submit ARQ enqueue; re-exports the retake
  policy from :mod:`.retake`.
* :mod:`.generation` — ARQ entrypoint that delegates to the T6.10
  pipeline.
* :mod:`.evaluation` — ARQ entrypoint that composes the T6.8 evaluation
  + T6.9 gap-report stages and persists their output.

Import-linter contract #1 forbids these modules from importing
``sqlalchemy`` at module level. Each sub-module declares
``AsyncSession`` under :class:`typing.TYPE_CHECKING` only and delegates
all DB access to the queries layer. INSERT/UPDATE flows construct ORM
instances via ``db.add(instance)`` / ``await db.flush()`` — those are
``AsyncSession`` methods, not module-level ``sqlalchemy`` imports.
"""

from __future__ import annotations

from abridgeai.features.interviews.services import (
    authoring,
    evaluation,
    generation,
    taking,
)

__all__ = ["authoring", "evaluation", "generation", "taking"]
