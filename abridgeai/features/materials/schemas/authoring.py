"""Teacher-facing (authoring) DTOs for the materials feature (T4.2).

Each Authoring schema inherits from its Public counterpart and widens /
adds fields that are only safe to expose to material authors:

* widened processing-state visibility (the full 9-value
  :data:`~abridgeai.features.materials.schemas.status.ProcessingStatusLiteral`
  rather than implicit ``"ready"``);
* audit columns from :class:`~abridgeai.core.db.AuditedByMixin` and
  :class:`~abridgeai.core.db.SoftDeleteMixin` (``created_by`` /
  ``updated_by`` / ``deleted_at`` / ``deleted_by``);
* version metadata (``current_version_id`` plus a flattened
  :class:`MaterialVersionAuthoring` slice for the latest revision);
* the ``visible_to_students`` toggle and ``ai_processing_enabled``
  flag — both teacher-only knobs from the baseline DDL.

Field drops vs plan body §4710 (per T4.1 ORM ground truth, §A13)
----------------------------------------------------------------
The plan body suggested several fields that do NOT exist in the
T4.1 ORM (which itself mirrors the baseline DDL, §C10):

* :class:`MaterialAuthoring`: ``description``, ``position``,
  ``internal_notes`` — none of these are columns on
  :class:`~abridgeai.features.materials.models.LearningMaterial`.
  ``LearningMaterial`` carries only ``lesson_id`` / ``title`` /
  ``material_type`` / ``ai_processing_enabled`` / ``visible_to_students``
  / ``current_version_id`` plus the audit + soft-delete column set.
* :class:`MaterialVersionAuthoring`: ``checksum_sha256`` lives on the
  ``storage_objects`` row, not on ``learning_material_versions``
  (Reconciliation §C14 — storage-object metadata is owned by the
  storage feature, not duplicated onto the version row). It surfaces
  via the request-side :class:`MaterialUploadComplete` only.

If a future migration adds any of these columns, this module is the
only place that needs updating (Public schemas continue to hide them
automatically).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ConfigDict

from abridgeai.features.materials.schemas.public import (
    MaterialPublic,
    _ORMModel,
)
from abridgeai.features.materials.schemas.status import ProcessingStatusLiteral


class MaterialVersionAuthoring(_ORMModel):
    """Teacher-facing projection of one ``LearningMaterialVersion`` row.

    Carries the full processing-state column set plus audit metadata.
    ``storage_object_id`` is non-null in the ORM (NOT NULL + RESTRICT
    FK to ``storage_objects``); the schema mirrors that.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    material_id: UUID
    version_no: int
    storage_object_id: UUID
    is_current: bool
    processing_status: ProcessingStatusLiteral
    processing_error: str | None = None
    extracted_metadata: dict[str, Any] = {}
    uploaded_by: UUID | None = None
    uploaded_at: datetime
    processed_at: datetime | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


class MaterialAuthoring(MaterialPublic):
    """Teacher-facing material DTO. Inherits public fields, exposes authoring extras.

    The ``latest_version`` slice is composed by the service layer (T4.5)
    and may be ``None`` for a freshly-created material that has no
    version row yet (matches ``LearningMaterial.current_version_id``
    being nullable in the ORM).
    """

    ai_processing_enabled: bool = True
    visible_to_students: bool = True
    current_version_id: UUID | None = None
    version_count: int = 0
    latest_version: MaterialVersionAuthoring | None = None
    created_by: UUID | None = None
    updated_by: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by: UUID | None = None


__all__ = [
    "MaterialAuthoring",
    "MaterialVersionAuthoring",
]
