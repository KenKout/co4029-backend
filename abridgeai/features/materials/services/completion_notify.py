"""Teacher completion/failure notifications for material ingestion (T7.6).

When a teacher uploads a material and AI processing is enabled, the actual
ingest runs asynchronously in an ARQ worker (``ingest_material_version_task``)
and can take seconds to minutes. Without a signal the teacher has no idea when
the document is ready (visible/previewable to students) or that it failed.

This module builds and dispatches a ``material_processing`` notification —
success OR failure — to the teacher who initiated the ingest, via the existing
``send_notification`` surface (in-app always-on; email preference-gated + async).

Design notes
------------
* Best-effort: every entrypoint swallows its own errors. A notification write
  must NEVER fail (or roll back) the ingest transaction it rides on — same
  contract as the SR remediation dispatcher.
* The recipient is the ``actor_id`` the worker was invoked with (the teacher
  who kicked off the upload/reprocess).
* ``action_url`` is the TEACHER lesson page (Option B: full routing context
  baked in at creation time), so "Take action" jumps straight to the AI Hub /
  Downloadable Resources for that lesson.
* Cross-feature Postgres reads go through raw ``text(...)`` — no foreign
  feature ORM imports, keeping the import-linter contracts satisfied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from abridgeai.core.observability import get_logger
from abridgeai.features.notifications.services.dispatch import send_notification

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

_CATEGORY = "material_processing"
_ENTITY_TYPE = "material"


async def _resolve_material_context(
    db: AsyncSession, material_version_id: UUID
) -> dict[str, object] | None:
    """Resolve version → material → lesson → module → course for the deep link.

    Returns a dict with ``material_id``, ``material_title``, ``lesson_id`` and
    ``course_id`` (all as UUIDs / str), or ``None`` when the row can't be found
    (deleted mid-flight, etc.) so the caller can skip silently.
    """
    row = (
        await db.execute(
            text(
                """
                SELECT
                    lm.id         AS material_id,
                    lm.title      AS material_title,
                    l.id          AS lesson_id,
                    m.course_id   AS course_id
                FROM learning_material_versions lmv
                JOIN learning_materials lm ON lm.id = lmv.material_id
                JOIN lessons l            ON l.id = lm.lesson_id
                JOIN modules m            ON m.id = l.module_id
                WHERE lmv.id = :version_id
                """
            ),
            {"version_id": str(material_version_id)},
        )
    ).mappings().first()
    if row is None:
        return None
    return dict(row)


async def notify_material_processing_outcome(
    db: AsyncSession,
    *,
    recipient_user_id: UUID,
    material_version_id: UUID,
    succeeded: bool,
    error_message: str | None = None,
    arq_pool: object | None = None,
) -> None:
    """Dispatch a ``material_processing`` notification to the initiating teacher.

    Best-effort — any failure is logged and swallowed so it can never break the
    ingest transaction this rides on.
    """
    try:
        ctx = await _resolve_material_context(db, material_version_id)
        if ctx is None:
            _logger.warning(
                "material_notify_skipped_missing_context",
                material_version_id=str(material_version_id),
            )
            return

        material_title = str(ctx["material_title"] or "Material")
        lesson_id = ctx["lesson_id"]
        course_id = ctx["course_id"]
        action_url = f"/teacher/courses/{course_id}/lessons/{lesson_id}"

        if succeeded:
            title = "Material ready"
            body = (
                f'"{material_title}" finished processing and is ready to show '
                "to students."
            )
        else:
            title = "Material processing failed"
            detail = f" ({error_message})" if error_message else ""
            body = (
                f'"{material_title}" could not be processed{detail}. '
                "Open the lesson to retry."
            )

        await send_notification(
            db,
            recipient_user_id=recipient_user_id,
            notification_type=_CATEGORY,
            title=title,
            body=body,
            entity_type=_ENTITY_TYPE,
            entity_id=(
                UUID(str(ctx["material_id"]))
                if not isinstance(ctx["material_id"], UUID)
                else ctx["material_id"]
            ),
            action_url=action_url,
            arq_pool=arq_pool,
        )
    except Exception:  # noqa: BLE001 — best-effort; never break the caller.
        _logger.exception(
            "material_notify_failed",
            material_version_id=str(material_version_id),
            succeeded=succeeded,
        )
        # Swallowing the exception is not enough: if the failure happened at
        # flush (e.g. the recipient user was deleted between upload and
        # completion), the session is left in pending-rollback and the
        # CALLER's next ``db.commit()`` raises — so a failed courtesy
        # notification retro-failed an ingest that had already committed, and
        # ARQ re-ran the whole pipeline. Roll back to leave the session clean.
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001 — a dead connection; caller's commit will surface it
            _logger.exception(
                "material_notify_rollback_failed",
                material_version_id=str(material_version_id),
            )


__all__ = ["notify_material_processing_outcome"]
