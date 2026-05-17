"""Student-facing (public) DTOs for the materials feature (T4.2).

Backed by the learner-facing slice of the materials router (T4.6). The
public schemas represent the narrowest projection — only ready, visible
materials and their presigned stream URLs surface here. Internal columns
(``processing_error``, audit / soft-delete metadata, ``uploaded_by``)
are gated behind :mod:`.authoring` and never reach the public API.

Reconciliation directives (HIGHER PRECEDENCE — see plan §266-585):

* §C10 — ``material_type`` is VARCHAR + CHECK with the 9-value set
  ``video, pdf, code, audio, image, docx, pptx, xlsx, text`` (per
  T4.1's baseline DDL). The ``Literal`` here mirrors the CHECK
  constraint byte-for-byte so a schema-level ValidationError fires
  before Postgres ever sees an invalid value.
* §C15 — :class:`MaterialPublic` does NOT expose ``processing_status``;
  the visibility filter at the query layer (T4.3) constrains the public
  catalog to versions where ``is_current = TRUE AND processing_status =
  'ready'``. Carrying ``processing_status`` in the public DTO would
  duplicate that gate at the wrong layer.

Field drops vs plan body §4707-4708 (per T4.1 ORM ground truth, §A13)
---------------------------------------------------------------------
The plan body lists ``description`` and ``position`` on
:class:`MaterialPublic`, but the baseline ``learning_materials`` DDL
declares neither column (lesson-scoped ordering lives on the
``ModuleItem.position`` of the parent lesson, not on the material
itself; descriptions are not modelled at all). Adding either field
would violate §A13 ("baseline > spec"), so both are dropped here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

MaterialTypeLiteral = Literal[
    "video",
    "pdf",
    "code",
    "audio",
    "image",
    "docx",
    "pptx",
    "xlsx",
    "text",
]


class _ORMModel(BaseModel):
    """Shared base — Pydantic v2 ORM-mode equivalent (`from_attributes=True`)."""

    model_config = ConfigDict(from_attributes=True)


class MaterialPublic(_ORMModel):
    """Student-facing material summary.

    Surfaces only what the learner needs: identity, lesson linkage,
    title, and the material's content type. Processing state, version
    metadata, and audit columns are gated to :mod:`.authoring`.
    """

    id: UUID
    lesson_id: UUID
    title: str
    material_type: MaterialTypeLiteral


class MaterialStreamUrl(_ORMModel):
    """Presigned URL response for a learner streaming a ready material.

    The TTL is decided by the backend (plan §4719 — "do NOT bake
    presigned URL TTL into schema"); the schema only carries the
    resolved ``expires_at`` for the client to know when to re-fetch.
    """

    url: str
    expires_at: datetime


__all__ = [
    "MaterialPublic",
    "MaterialStreamUrl",
    "MaterialTypeLiteral",
]
