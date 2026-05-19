"""Teacher-side authoring service for the materials feature (T4.5).

Owns the direct-upload lifecycle (single-shot + multipart), version
register-on-complete with mandatory ``head_object`` verification (the
phantom-complete attack guard from plan §4934 + Reconciliation §C9),
soft-delete that preserves the underlying S3 object (plan §4946 +
§4954 — recovery requires retention), and reprocess with a chunk
purge + concurrency check (Reconciliation §C13).

Architectural rules honoured:

* ``services -> sqlalchemy`` import-linter contract (T0.4) — submodules
  reference :class:`AsyncSession` only under :data:`TYPE_CHECKING`.
  Runtime data access goes through ``queries.*``; the rare write
  path that needs raw ORM (``DocumentChunk`` purge on reprocess) +
  ``storage_objects`` writes use the localised raw-SQL escape hatch in
  :mod:`._storage`, mirroring the original module's documented pattern.
* Routers→services boundary — every router endpoint composes a helper
  re-exported here; no direct ``queries.*`` access from
  :mod:`features.materials.routers.authoring`.
* Service layer flushes; the router commits. ARQ jobs are enqueued
  AFTER the flush so the version + processing_job rows are visible
  in DB once the job dequeues.

Package layout (DoD §136 — 800 LOC review cap)
----------------------------------------------
The original ``authoring.py`` (953 LOC) was split into focused modules.
All public symbols are re-exported here so existing import sites
(``from abridgeai.features.materials.services.authoring import …``)
keep working unchanged:

* :mod:`._common` — DTOs, MIME table, ``_require_*`` / ``_resolve_*``
  helpers, service-layer exceptions, ``resolve_course_id_for_material``.
* :mod:`._storage` — raw-SQL writers for ``storage_objects`` +
  version-state mutations.
* :mod:`._reads` — read-side composition (list, get, update, progress).
* :mod:`._upload` — direct-upload init + multipart presign / complete /
  abort.
* :mod:`._versions` — ``head_object``-verified ``complete_upload``,
  reprocess, soft-delete.

The :func:`get_settings` re-export is intentional: tests monkeypatch
``abridgeai.features.materials.services.authoring.get_settings`` and
the upload module looks the symbol up via this package so the override
remains effective post-split.
"""

from __future__ import annotations

from abridgeai.core.config import get_settings
from abridgeai.features.materials.services.authoring._common import (
    CompletedPartIn,
    ConcurrentReprocessError,
    HeadVerificationError,
    MaterialUploadInitResponse,
    MultipartPartsResponse,
    ReprocessResponse,
    UploadCompleteResponse,
    resolve_course_id_for_material,
)
from abridgeai.features.materials.services.authoring._reads import (
    get_authoring_material,
    get_authoring_stream_url,
    get_lesson_processing_summary_view,
    get_processing_progress,
    list_authoring_materials,
    update_material,
)
from abridgeai.features.materials.services.authoring._upload import (
    abort_multipart,
    complete_multipart,
    fetch_multipart_parts,
    init_upload,
)
from abridgeai.features.materials.services.authoring._versions import (
    complete_upload,
    reprocess_material,
    soft_delete_material,
)

__all__ = [
    "CompletedPartIn",
    "ConcurrentReprocessError",
    "HeadVerificationError",
    "MaterialUploadInitResponse",
    "MultipartPartsResponse",
    "ReprocessResponse",
    "UploadCompleteResponse",
    "abort_multipart",
    "complete_multipart",
    "complete_upload",
    "fetch_multipart_parts",
    "get_authoring_material",
    "get_authoring_stream_url",
    "get_lesson_processing_summary_view",
    "get_processing_progress",
    "get_settings",
    "init_upload",
    "list_authoring_materials",
    "reprocess_material",
    "resolve_course_id_for_material",
    "soft_delete_material",
    "update_material",
]
