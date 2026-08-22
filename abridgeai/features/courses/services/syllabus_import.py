"""Turn an uploaded course syllabus into a draft course.

One manager-facing operation, run synchronously inside the request:

1. parse the PDF (:mod:`features.courses.ingest.syllabus`) in the language
   the manager picked — pure, no side effects, so a bad file costs nothing;
2. create the course as a **draft**, owned by the importing manager, in
   their primary organization, with a slug de-duplicated against the org;
3. create the ``L.O.x.y`` tree as ``course_learning_outcomes`` rows;
4. archive the uploaded PDF to object storage so teachers and students can
   download the original syllabus later;
5. record the attempt in ``course_syllabus_imports`` and notify the manager
   — success (with any parser warnings) or failure (with the reason).

Why synchronous rather than an ARQ job like material ingestion: parsing is
deterministic regex work over already-extracted text with no LLM call and
no network hop, so it finishes inside a normal request. A background job
would buy nothing and cost a worker round-trip, a polling endpoint and a
"pending forever" failure mode. The manager still gets the notification
they asked for — it is what makes the result durable and what carries the
failure reason once the upload dialog is closed.

Transaction shape
-----------------
The success path is one transaction: course + outcomes + storage row +
import row commit together, so there is never a course without its
syllabus or an import row pointing at a course that was rolled back. Any
failure rolls that back and then writes the failure row on its own, which
is why :func:`_record_failure` runs after an explicit ``rollback()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import AppError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import CurrentUser
from abridgeai.features.courses.ingest import (
    ParsedSyllabus,
    SyllabusParseError,
    parse_syllabus_pdf,
)
from abridgeai.features.courses.ingest.syllabus import SyllabusLanguage
from abridgeai.features.courses.models import (
    Course,
    CourseLearningOutcome,
    CourseSyllabusImport,
)
from abridgeai.features.courses.queries import (
    authoring as authoring_queries,
)
from abridgeai.features.courses.queries import (
    get_user_primary_organization_id,
)
from abridgeai.features.courses.schemas import SyllabusImportResult, SyllabusImportRow
from abridgeai.infrastructure.s3 import create_stream_url, put_object_bytes

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)


@dataclass
class _StorageTarget:
    """Minimal ``bucket``/``object_key`` pair the s3 helpers accept.

    Declared here rather than imported from ``services.authoring`` so this
    module does not depend on a sibling's private name.
    """

    bucket: str
    object_key: str


# Only PDF for now: the parser reads a PDF text layer, and accepting .docx
# here would mean silently failing on every upload that is not a PDF anyway.
_ALLOWED_MIME_TYPES = frozenset({"application/pdf"})
_MAX_SYLLABUS_BYTES = 20 * 1024 * 1024  # 20 MiB — a text-layer syllabus is ~250 KiB.


class SyllabusImportError(ValueError):
    """A syllabus import that produced no course.

    Carries the user-facing ``code: sentence`` reason. The router maps it
    to HTTP 422; the same string is stored on the attempt row and copied
    into the manager's failure notification, so all three agree.
    """


async def import_course_from_syllabus(
    db: AsyncSession,
    *,
    data: bytes,
    content_type: str | None,
    filename: str | None,
    language: SyllabusLanguage,
    actor: CurrentUser,
    arq_pool: object | None = None,
) -> SyllabusImportResult:
    """Create a draft course from ``data``, or raise :class:`SyllabusImportError`.

    Every failure path — rejected upload, unparseable PDF, storage or DB
    error — records an attempt row and notifies ``actor`` before raising,
    so a manager who closed the dialog still learns what happened.
    """
    org_id = await get_user_primary_organization_id(db, actor.user_id)
    if org_id is None:
        raise AppError(f"User {actor.user_id} has no primary organization; cannot import a course.")

    try:
        _validate_upload(data, content_type)
        parsed = parse_syllabus_pdf(data, language)
    except (SyllabusImportError, SyllabusParseError) as exc:
        await _record_failure(
            db,
            organization_id=org_id,
            actor=actor,
            language=language,
            filename=filename,
            reason=str(exc),
            arq_pool=arq_pool,
        )
        raise SyllabusImportError(str(exc)) from exc

    try:
        return await _build_course(
            db,
            parsed=parsed,
            data=data,
            filename=filename,
            language=language,
            organization_id=org_id,
            actor=actor,
            arq_pool=arq_pool,
        )
    except Exception as exc:
        # The parse succeeded, so this is a storage/DB problem rather than a
        # bad document. Roll the half-built course back before writing the
        # attempt row — otherwise the failure record would ride the same
        # doomed transaction and vanish with it.
        await db.rollback()
        _logger.exception(
            "syllabus_import_build_failed",
            user_id=str(actor.user_id),
            organization_id=str(org_id),
            filename=filename,
        )
        reason = _build_failure_reason(exc)
        await _record_failure(
            db,
            organization_id=org_id,
            actor=actor,
            language=language,
            filename=filename,
            reason=reason,
            arq_pool=arq_pool,
        )
        raise SyllabusImportError(reason) from exc


def _build_failure_reason(exc: Exception) -> str:
    """A failure message that names the ACTUAL fault, not just its class."""
    origin = getattr(exc, "orig", None)
    detail = str(origin if origin is not None else exc).strip()
    first_line = detail.splitlines()[0].strip() if detail else ""
    suffix = f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__
    return (
        "import_failed: the syllabus was read successfully but the course "
        f"could not be created ({suffix[:400]})."
    )


def _validate_upload(data: bytes, content_type: str | None) -> None:
    """Reject uploads the parser could not possibly read, with a clear reason."""
    if not data:
        raise SyllabusImportError("empty_file: the uploaded file is empty.")
    if len(data) > _MAX_SYLLABUS_BYTES:
        raise SyllabusImportError("syllabus_too_large: the file must be 20 MiB or smaller.")
    # Browsers occasionally send application/octet-stream for a drag-and-drop
    # PDF, so the magic bytes are the authority and the header is only used
    # when it positively disagrees.
    if content_type and content_type.split(";")[0].strip() not in _ALLOWED_MIME_TYPES:
        raise SyllabusImportError(
            "unsupported_syllabus_type: only PDF syllabus files are supported."
        )
    if not data.startswith(b"%PDF-"):
        raise SyllabusImportError(
            "unsupported_syllabus_type: only PDF syllabus files are supported."
        )


async def _build_course(
    db: AsyncSession,
    *,
    parsed: ParsedSyllabus,
    data: bytes,
    filename: str | None,
    language: SyllabusLanguage,
    organization_id: UUID,
    actor: CurrentUser,
    arq_pool: object | None,
) -> SyllabusImportResult:
    """The success path, all in one transaction."""
    slug = await _unique_slug(
        db, organization_id=organization_id, preferred=parsed.suggested_slug()
    )

    course = Course(
        organization_id=organization_id,
        owner_user_id=actor.user_id,
        slug=slug,
        title=parsed.title,
        description=parsed.description,
        # Imported courses always land as drafts: the syllabus gives the
        # shell (title, description, outcomes) but no modules, lessons or
        # assessments, so there is nothing a learner could do with it yet.
        status="draft",
        estimated_minutes=parsed.estimated_minutes,
        created_by=actor.user_id,
        updated_by=actor.user_id,
    )
    db.add(course)
    await db.flush()

    outcome_count = _create_outcomes(db, parsed, course_id=course.id, actor_id=actor.user_id)
    await db.flush()

    storage_object_id = await _archive_syllabus(
        db,
        data=data,
        filename=filename,
        course_id=course.id,
        uploaded_by=actor.user_id,
    )

    record = CourseSyllabusImport(
        organization_id=organization_id,
        course_id=course.id,
        storage_object_id=storage_object_id,
        imported_by=actor.user_id,
        language=language,
        status="succeeded",
        original_filename=(filename or None),
        warnings=list(parsed.warnings),
        outcome_count=outcome_count,
    )
    db.add(record)
    await db.flush()

    await _notify_module().notify_syllabus_import_succeeded(
        db,
        manager_user_id=actor.user_id,
        course_id=course.id,
        course_title=course.title,
        outcome_count=outcome_count,
        warnings=list(parsed.warnings),
        arq_pool=arq_pool,
    )
    await db.commit()

    return SyllabusImportResult(
        import_id=record.id,
        course_id=course.id,
        course_slug=course.slug,
        title=course.title,
        language=language,
        description=course.description,
        estimated_minutes=course.estimated_minutes,
        outcome_count=outcome_count,
        warnings=list(parsed.warnings),
    )


def _create_outcomes(
    db: AsyncSession, parsed: ParsedSyllabus, *, course_id: UUID, actor_id: UUID
) -> int:
    """Insert the ``L.O.`` tree, parents before children.

    Source codes (``"1"``, ``"1.3"``) only decide the SHAPE — positions are
    assigned as consecutive sibling ranks in document order, because the
    displayed code is re-derived from sibling order and a stored gap would
    not survive the round trip anyway. The parser already warned the
    manager when the source numbering had a gap.

    A child whose parent code never appeared is attached at the top level
    rather than dropped: a malformed syllabus should lose structure, not
    content.
    """
    # ``id`` is a SERVER-side default (``uuid_generate_v4()``), so an unflushed
    # row has no id yet and a child could not reference its parent. Minting the
    # UUID here — the same thing the thumbnail upload does for its storage row —
    # lets the whole tree be added and flushed in one go instead of a
    # flush-per-level.
    ids: dict[str, UUID] = {}
    next_position: dict[UUID | None, int] = {}

    # Shallowest first so a parent id always exists by the time its child is
    # built, regardless of the order the codes appeared in the document.
    ordered = sorted(parsed.outcomes, key=lambda o: o.code.count("."))
    for item in ordered:
        parent_id = ids.get(item.parent_code) if item.parent_code else None
        position = next_position.get(parent_id, 0) + 1
        next_position[parent_id] = position

        outcome_id = uuid4()
        ids[item.code] = outcome_id
        db.add(
            CourseLearningOutcome(
                id=outcome_id,
                course_id=course_id,
                parent_id=parent_id,
                position=position,
                outcome_text=item.text,
                created_by=actor_id,
                updated_by=actor_id,
            )
        )

    return len(ids)


async def _archive_syllabus(
    db: AsyncSession,
    *,
    data: bytes,
    filename: str | None,
    course_id: UUID,
    uploaded_by: UUID,
) -> UUID:
    """Store the source PDF and return its ``storage_objects.id``.

    Uploaded before the DB row is added, so a storage failure aborts the
    whole import instead of leaving a row pointing at an object that was
    never written.
    """
    settings = get_settings()
    bucket = settings.s3_bucket_name or "abridgeai-local"
    object_id = uuid4()
    object_key = f"course-syllabi/{course_id}/{object_id}.pdf"

    await put_object_bytes(
        _StorageTarget(bucket=bucket, object_key=object_key),
        data,
        content_type="application/pdf",
    )

    authoring_queries.insert_storage_object(
        db,
        object_id=object_id,
        bucket=bucket,
        object_key=object_key,
        original_filename=(filename or "syllabus.pdf")[:255],
        mime_type="application/pdf",
        size_bytes=len(data),
        uploaded_by=uploaded_by,
        uploaded_at=datetime.now(tz=UTC),
    )
    return object_id


async def _unique_slug(db: AsyncSession, *, organization_id: UUID, preferred: str) -> str:
    """``preferred``, or ``preferred-2`` / ``-3`` … when the org already uses it.

    One query fetches every colliding slug so the suffix search is in
    memory; probing candidates one at a time would be a round-trip per
    collision and still racy. The real guarantee is the
    ``uq_courses_org_slug`` unique index — a concurrent importer picking
    the same suffix fails there and the caller reports it as an import
    failure rather than writing a duplicate.
    """
    taken = await authoring_queries.course_slugs_with_prefix(
        db, organization_id=organization_id, prefix=preferred
    )
    if preferred not in taken:
        return preferred
    suffix = 2
    while f"{preferred}-{suffix}"[:100] in taken:
        suffix += 1
    return f"{preferred}-{suffix}"[:100]


async def _record_failure(
    db: AsyncSession,
    *,
    organization_id: UUID,
    actor: CurrentUser,
    language: SyllabusLanguage,
    filename: str | None,
    reason: str,
    arq_pool: object | None,
) -> None:
    """Persist a failed attempt and notify the manager, best effort.

    Commits on its own: the caller is about to raise, and a failure the
    manager was told about must not disappear with the request. If even
    this write fails there is nothing useful left to do but log — the
    manager still gets the HTTP error.
    """
    try:
        record = CourseSyllabusImport(
            organization_id=organization_id,
            course_id=None,
            storage_object_id=None,
            imported_by=actor.user_id,
            language=language,
            status="failed",
            original_filename=(filename or None),
            error_message=reason,
            warnings=[],
            outcome_count=0,
        )
        db.add(record)
        await db.flush()
        await _notify_module().notify_syllabus_import_failed(
            db,
            manager_user_id=actor.user_id,
            import_id=record.id,
            filename=filename,
            reason=reason,
            arq_pool=arq_pool,
        )
        await db.commit()
    except Exception:  # noqa: BLE001 — the caller is already raising the real error
        _logger.exception(
            "syllabus_import_failure_record_failed",
            user_id=str(actor.user_id),
            filename=filename,
        )
        await db.rollback()


def _notify_module() -> ModuleType:
    """The sibling ``notify`` module, imported lazily.

    Same reason as its other callers in this feature: a module-level
    ``courses.services -> notify`` edge closes an import cycle through
    ``enrollments``.
    """
    from abridgeai.features.courses.services import notify  # noqa: PLC0415

    return notify


async def list_syllabus_imports(
    db: AsyncSession, *, actor: CurrentUser, limit: int = 50
) -> list[SyllabusImportRow]:
    """Recent import attempts in the actor's org, newest first."""
    org_id = await get_user_primary_organization_id(db, actor.user_id)
    if org_id is None:
        return []
    rows = await authoring_queries.list_syllabus_imports(db, organization_id=org_id, limit=limit)
    return [SyllabusImportRow.model_validate(row) for row in rows]


async def get_syllabus_download_url(db: AsyncSession, course_id: UUID) -> tuple[str, datetime]:
    """Presigned GET URL for a course's archived syllabus.

    Unlike the thumbnail helper this propagates storage errors instead of
    returning ``None``: the caller asked for this specific file by name, so
    a blip has to surface as an error rather than a silent "no syllabus".
    """
    target = await authoring_queries.get_syllabus_storage_target(db, course_id)
    if target is None:
        raise SyllabusImportError("no_syllabus: this course has no imported syllabus document.")
    bucket, object_key, _filename = target
    url, expires_at = await create_stream_url(_StorageTarget(bucket=bucket, object_key=object_key))
    return url, expires_at


async def course_has_syllabus(db: AsyncSession, course_id: UUID) -> bool:
    """Whether a download link should be offered for this course at all."""
    return await authoring_queries.get_latest_syllabus_import(db, course_id) is not None


__all__ = [
    "SyllabusImportError",
    "course_has_syllabus",
    "get_syllabus_download_url",
    "import_course_from_syllabus",
    "list_syllabus_imports",
]
