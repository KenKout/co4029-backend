"""Business rules for policy authoring and reading.

The state machine mirrors ``career_paths.services.authoring``: a version is a
draft while it is written and frozen once published. The two rules that matter:

1. **A published version is never edited.** Editing means opening a new draft
   (copy-on-write from the latest version) and publishing that. Without this,
   correcting a typo silently rewrites the document a reader already agreed
   to, and the version number stops meaning anything.
2. **At most one open draft per (policy, language).** A second draft makes
   "the draft" ambiguous for both the editor and the publish action.

Bodies are sanitized on every write with the SAME nh3 allow-list the quiz
rich-content path uses, so there is one place to audit rather than two.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.sanitize import sanitize_rich_content
from abridgeai.features.access_control.api import public as access_api
from abridgeai.features.identity.api import public as identity_api
from abridgeai.features.policies import queries as policy_queries
from abridgeai.features.policies.models import Policy, PolicyVersion
from abridgeai.features.policies.schemas import (
    PolicyAudienceRoleRead,
    PolicyAudienceUpdate,
    PolicyCreate,
    PolicyDetail,
    PolicyDocument,
    PolicySummary,
    PolicyVersionCreate,
    PolicyVersionPatch,
    PolicyVersionRead,
    PolicyVersionSummary,
)

if TYPE_CHECKING:
    from abridgeai.core.db import AsyncSession  # type: ignore[attr-defined]

#: The only body format today. Declared rather than inlined so the sanitizer
#: call and the stored discriminator cannot drift apart.
BODY_FORMAT = "markdown"

DEFAULT_LANGUAGE = "en"


def _clean(body: str) -> str:
    cleaned = sanitize_rich_content(body, fmt=BODY_FORMAT)
    # sanitize_rich_content is typed to return None only for a None input.
    return cleaned or ""


# ---------------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------------


async def create_policy(
    db: AsyncSession, payload: PolicyCreate, *, actor_id: UUID | None
) -> PolicyDetail:
    """Create a policy and open version 1 as a draft.

    A policy with no version at all would be unreachable and unopenable in the
    editor, so creation always yields something to write in.
    """
    existing = await policy_queries.get_policy_by_slug(db, payload.slug)
    if existing is not None:
        raise ConflictError(f"A policy with slug '{payload.slug}' already exists")

    policy = await policy_queries.insert_policy(
        db, slug=payload.slug, category=payload.category, actor_id=actor_id
    )
    await policy_queries.insert_version(
        db,
        policy_id=policy.id,
        version_no=1,
        language=payload.language,
        title=payload.title,
        body="",
        changelog=None,
        actor_id=actor_id,
    )
    return await policy_detail(db, policy.id)


async def policy_detail(db: AsyncSession, policy_id: UUID) -> PolicyDetail:
    policy = await policy_queries.get_policy(db, policy_id)
    if policy is None or policy.deleted_at is not None:
        raise NotFoundError("Policy not found")

    versions = await policy_queries.list_versions(db, policy_id)
    role_ids = set(await policy_queries.audience_role_ids(db, policy_id))
    # Names come from the roles catalogue rather than being stored alongside
    # the audience, so a renamed role is renamed everywhere at once.
    roles = [r for r in await access_api.list_roles(db) if r.id in role_ids]
    return PolicyDetail(
        id=policy.id,
        slug=policy.slug,
        category=policy.category,  # type: ignore[arg-type]
        audience=[PolicyAudienceRoleRead(role_id=r.id, code=r.code, name=r.name) for r in roles],
        versions=[PolicyVersionSummary.model_validate(v) for v in versions],
    )


async def list_policies(db: AsyncSession) -> list[PolicyDetail]:
    rows = await policy_queries.list_policies(db)
    return [await policy_detail(db, row.id) for row in rows]


async def open_new_draft(
    db: AsyncSession,
    policy_id: UUID,
    payload: PolicyVersionCreate,
    *,
    actor_id: UUID | None,
) -> PolicyVersionSummary:
    """Open the next draft, seeded from the latest version in that language.

    Copy-on-write: an admin fixing one clause should start from the text that
    is live, not from an empty box.
    """
    policy = await policy_queries.get_policy(db, policy_id)
    if policy is None or policy.deleted_at is not None:
        raise NotFoundError("Policy not found")

    language = payload.language
    existing_draft = await policy_queries.open_draft(db, policy_id, language=language)
    if existing_draft is not None:
        raise ConflictError(
            f"A draft is already open for this policy in '{language}'; "
            "edit or publish it before opening another"
        )

    latest = await policy_queries.latest_version(db, policy_id, language=language)
    if latest is None and payload.title is None:
        raise AppError("The first version in a language needs a title")

    version = await policy_queries.insert_version(
        db,
        policy_id=policy_id,
        version_no=(latest.version_no + 1) if latest else 1,
        language=language,
        title=payload.title or (latest.title if latest else ""),
        body=_clean(payload.body if payload.body is not None else (latest.body if latest else "")),
        changelog=payload.changelog,
        actor_id=actor_id,
    )
    return PolicyVersionSummary.model_validate(version)


async def _editable_draft(db: AsyncSession, version_id: UUID) -> PolicyVersion:
    version = await policy_queries.get_version(db, version_id)
    if version is None or version.deleted_at is not None:
        raise NotFoundError("Policy version not found")
    if version.status != "draft":
        raise ConflictError("A published version is frozen. Open a new draft to change the text.")
    return version


async def update_draft(
    db: AsyncSession,
    version_id: UUID,
    payload: PolicyVersionPatch,
    *,
    actor_id: UUID | None,
) -> PolicyVersionSummary:
    version = await _editable_draft(db, version_id)
    fields = payload.model_dump(exclude_unset=True)

    if "title" in fields and fields["title"] is not None:
        version.title = fields["title"]
    if "body" in fields and fields["body"] is not None:
        version.body = _clean(fields["body"])
    if "changelog" in fields:
        version.changelog = fields["changelog"]
    version.updated_by = actor_id

    await db.flush()
    # ``updated_at`` is a SERVER onupdate (TimestampMixin): the flush pushes the
    # UPDATE but leaves the column unloaded on the instance, and reading it
    # during response validation triggers a synchronous lazy load — which on an
    # AsyncSession raises MissingGreenlet (the 500 every draft save hit). An
    # explicit await refresh loads it greenlet-safely.
    await db.refresh(version, attribute_names=["updated_at"])
    return PolicyVersionSummary.model_validate(version)


async def read_version(db: AsyncSession, policy_id: UUID, version_id: UUID) -> PolicyVersionRead:
    """One version WITH its body — what the authoring editor loads.

    ``PolicyDetail`` deliberately carries version summaries only, so a policy
    with a long history is not a large response. The editor needs exactly one
    body, so it asks for exactly one.

    ``policy_id`` is verified against the version rather than ignored: the URL
    asserts a parent, and an id pair that does not actually match is a bug on
    the caller's side, not a document to serve.
    """
    version = await policy_queries.get_version(db, version_id)
    if version is None or version.deleted_at is not None or version.policy_id != policy_id:
        raise NotFoundError("Policy version not found")
    return PolicyVersionRead.model_validate(version)


async def publish_version(
    db: AsyncSession, version_id: UUID, *, actor_id: UUID | None
) -> PolicyVersionSummary:
    """Release a draft and retire whatever it supersedes.

    ``published_at`` / ``published_by`` are stamped here rather than accepted
    from the client, so the attribution is always the account that actually
    performed the release.
    """
    version = await _editable_draft(db, version_id)
    if not version.body.strip():
        raise AppError("A policy cannot be published with an empty body")

    version.status = "published"
    version.published_at = datetime.now(tz=UTC)
    version.published_by = actor_id
    version.updated_by = actor_id
    await db.flush()

    await policy_queries.supersede_published(
        db, version.policy_id, language=version.language, keep_id=version.id
    )
    await db.flush()
    # Same server-onupdate lazy-load trap as update_draft — see there.
    await db.refresh(version, attribute_names=["updated_at"])
    return PolicyVersionSummary.model_validate(version)


async def set_audience(
    db: AsyncSession,
    policy_id: UUID,
    payload: PolicyAudienceUpdate,
    *,
    actor_id: UUID | None,
) -> PolicyDetail:
    """Replace the audience set. An empty list makes the policy public."""
    policy = await policy_queries.get_policy(db, policy_id)
    if policy is None or policy.deleted_at is not None:
        raise NotFoundError("Policy not found")

    codes = list(dict.fromkeys(payload.role_codes))
    roles = await access_api.get_roles_by_codes(db, codes)
    unknown = [c for c in codes if c not in roles]
    if unknown:
        raise AppError(f"Unknown role code(s): {', '.join(sorted(unknown))}")

    await policy_queries.replace_audience(
        db, policy_id, role_ids=[roles[c].id for c in codes], actor_id=actor_id
    )
    return await policy_detail(db, policy_id)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _published_at(version: PolicyVersion) -> datetime:
    """Publication timestamp of a version that is known to be published.

    ``ck_policy_versions_published_at`` guarantees the pairing at the database
    level, so reaching the raise means the row was written around the schema —
    worth failing loudly rather than coercing to a fake date.
    """
    if version.published_at is None:
        raise AppError(f"Policy version {version.id} is published but has no publication date")
    return version.published_at


def _document(policy: Policy, version: PolicyVersion, publisher: str | None) -> PolicyDocument:
    return PolicyDocument(
        slug=policy.slug,
        category=policy.category,  # type: ignore[arg-type]
        title=version.title,
        body=version.body,
        format=version.format,
        language=version.language,
        version_no=version.version_no,
        published_at=_published_at(version),
        published_by_name=publisher,
        changelog=version.changelog,
    )


async def read_document(
    db: AsyncSession, slug: str, *, language: str = DEFAULT_LANGUAGE
) -> PolicyDocument:
    """One policy's current published text.

    Deliberately NOT audience-filtered. The audience scopes the index a reader
    browses; a policy reached by its own URL must open, because these links are
    shared, emailed and bookmarked, and the text is public either way. Hiding a
    document from the person it governs was never a security boundary.
    """
    policy = await policy_queries.get_policy_by_slug(db, slug)
    if policy is None or policy.deleted_at is not None:
        raise NotFoundError(f"Policy '{slug}' not found")

    version = await policy_queries.published_version(db, policy.id, language=language)
    if version is None and language != DEFAULT_LANGUAGE:
        # Fall back to English rather than 404 — a missing translation should
        # degrade to a readable document, not to nothing.
        version = await policy_queries.published_version(db, policy.id, language=DEFAULT_LANGUAGE)
    if version is None:
        raise NotFoundError(f"Policy '{slug}' has no published version")

    publisher = None
    if version.published_by is not None:
        user = await identity_api.get_user_by_id(db, version.published_by)
        publisher = user.display_name if user else None
    return _document(policy, version, publisher)


async def list_documents(
    db: AsyncSession,
    *,
    role_codes: list[str] | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> list[PolicySummary]:
    """The index, scoped to the reader's roles.

    ``role_codes=None`` (signed out) is the anonymous reader: treated as a
    STUDENT — the universal role. A policy is visible when it has no
    audience, names one of the reader's roles, or names the student role
    (everyone is a party to the student policies; a prospective student is
    exactly who reads the terms). An unrecognised code simply matches
    nothing rather than erroring: this is a courtesy filter over public
    documents, not a gate.
    """
    role_ids: list[UUID] | None = None
    if role_codes:
        roles = await access_api.get_roles_by_codes(db, role_codes)
        role_ids = [r.id for r in roles.values()]

    # The universal role: resolve once, every reader benefits. A role code
    # that does not exist in the catalogue (fresh deploy before seeding)
    # degrades to the plain public set rather than erroring.
    student_roles = await access_api.get_roles_by_codes(
        db, [policy_queries.STUDENT_ROLE_CODE]
    )
    student_role_id = next(iter(student_roles.values())).id if student_roles else None

    rows = await policy_queries.published_documents(
        db, language=language, role_ids=role_ids, universal_role_id=student_role_id
    )
    out: list[PolicySummary] = []
    for policy, version in rows:
        out.append(
            PolicySummary(
                slug=policy.slug,
                category=policy.category,  # type: ignore[arg-type]
                title=version.title,
                language=version.language,
                version_no=version.version_no,
                published_at=_published_at(version),
            )
        )
    return out


__all__ = [
    "BODY_FORMAT",
    "DEFAULT_LANGUAGE",
    "create_policy",
    "list_documents",
    "list_policies",
    "open_new_draft",
    "policy_detail",
    "publish_version",
    "read_version",
    "read_document",
    "set_audience",
    "update_draft",
]
