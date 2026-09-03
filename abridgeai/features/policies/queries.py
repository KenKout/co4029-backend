"""Data access for the policies feature.

Deliberately free of cross-feature model imports. Roles and users belong to
sibling features, and the import-linter "Features are independent" contract
routes that traffic through their ``api.public`` modules — so this layer deals
in role IDS and the service resolves them to names via
``access_control.api.public``.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.features.policies.models import Policy, PolicyAudienceRole, PolicyVersion


async def get_policy(db: AsyncSession, policy_id: UUID) -> Policy | None:
    return await db.get(Policy, policy_id)


async def get_policy_by_slug(db: AsyncSession, slug: str) -> Policy | None:
    stmt = select(Policy).where(Policy.slug == slug)
    return (await db.execute(stmt)).scalar_one_or_none()


async def list_policies(db: AsyncSession) -> list[Policy]:
    stmt = select(Policy).order_by(Policy.category, Policy.slug)
    return list((await db.execute(stmt)).scalars().all())


async def insert_policy(
    db: AsyncSession, *, slug: str, category: str, actor_id: UUID | None
) -> Policy:
    row = Policy(slug=slug, category=category, created_by=actor_id, updated_by=actor_id)
    db.add(row)
    await db.flush()
    return row


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


async def list_versions(db: AsyncSession, policy_id: UUID) -> list[PolicyVersion]:
    """Newest first per language, which is the order a history panel reads."""
    stmt = (
        select(PolicyVersion)
        .where(PolicyVersion.policy_id == policy_id)
        .order_by(PolicyVersion.language, PolicyVersion.version_no.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_version(db: AsyncSession, version_id: UUID) -> PolicyVersion | None:
    return await db.get(PolicyVersion, version_id)


async def latest_version(
    db: AsyncSession, policy_id: UUID, *, language: str
) -> PolicyVersion | None:
    """Highest ``version_no`` in this language, whatever its status.

    Used to seed a new draft (copy-on-write) and to compute the next number.
    """
    stmt = (
        select(PolicyVersion)
        .where(PolicyVersion.policy_id == policy_id, PolicyVersion.language == language)
        .order_by(PolicyVersion.version_no.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def open_draft(db: AsyncSession, policy_id: UUID, *, language: str) -> PolicyVersion | None:
    """The policy's existing draft in this language, if one is already open.

    At most one draft may exist per (policy, language): a second one would
    make "the draft" ambiguous for both the editor and the publish action.
    """
    stmt = select(PolicyVersion).where(
        PolicyVersion.policy_id == policy_id,
        PolicyVersion.language == language,
        PolicyVersion.status == "draft",
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def published_version(
    db: AsyncSession, policy_id: UUID, *, language: str
) -> PolicyVersion | None:
    """The current published version — highest published ``version_no``."""
    stmt = (
        select(PolicyVersion)
        .where(
            PolicyVersion.policy_id == policy_id,
            PolicyVersion.language == language,
            PolicyVersion.status == "published",
        )
        .order_by(PolicyVersion.version_no.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def insert_version(
    db: AsyncSession,
    *,
    policy_id: UUID,
    version_no: int,
    language: str,
    title: str,
    body: str,
    changelog: str | None,
    actor_id: UUID | None,
) -> PolicyVersion:
    row = PolicyVersion(
        policy_id=policy_id,
        version_no=version_no,
        language=language,
        status="draft",
        title=title,
        body=body,
        changelog=changelog,
        created_by=actor_id,
        updated_by=actor_id,
    )
    db.add(row)
    await db.flush()
    return row


async def supersede_published(
    db: AsyncSession, policy_id: UUID, *, language: str, keep_id: UUID
) -> None:
    """Archive every other published version in this language.

    Publishing v3 must retire v2 in the same transaction, or two rows both
    claim to be current and ``published_version`` silently picks one by
    ordering.
    """
    stmt = select(PolicyVersion).where(
        PolicyVersion.policy_id == policy_id,
        PolicyVersion.language == language,
        PolicyVersion.status == "published",
        PolicyVersion.id != keep_id,
    )
    for row in (await db.execute(stmt)).scalars().all():
        row.status = "archived"


# ---------------------------------------------------------------------------
# Audience
# ---------------------------------------------------------------------------


async def audience_role_ids(db: AsyncSession, policy_id: UUID) -> list[UUID]:
    """Role ids this policy names as a party.

    Ids only — the display name lives in the roles catalogue, which is another
    feature's data and is read through its public API.
    """
    stmt = select(PolicyAudienceRole.role_id).where(PolicyAudienceRole.policy_id == policy_id)
    return list((await db.execute(stmt)).scalars().all())


async def replace_audience(
    db: AsyncSession, policy_id: UUID, *, role_ids: Sequence[UUID], actor_id: UUID | None
) -> None:
    """Make the stored audience exactly ``role_ids``.

    Rows are added and removed rather than wiped and re-inserted so an
    unchanged role keeps its original ``created_at`` — the audit trail should
    say when a role was actually added to the policy, not when the set was
    last saved.
    """
    existing = {
        row.role_id: row
        for row in (
            await db.execute(
                select(PolicyAudienceRole).where(PolicyAudienceRole.policy_id == policy_id)
            )
        )
        .scalars()
        .all()
    }
    wanted = set(role_ids)

    for role_id, row in existing.items():
        if role_id not in wanted:
            await db.delete(row)

    for role_id in wanted - set(existing):
        db.add(
            PolicyAudienceRole(
                policy_id=policy_id,
                role_id=role_id,
                created_by=actor_id,
                updated_by=actor_id,
            )
        )
    await db.flush()


# ---------------------------------------------------------------------------
# Reader-facing reads
# ---------------------------------------------------------------------------


async def published_documents(
    db: AsyncSession, *, language: str, role_ids: Sequence[UUID] | None
) -> list[tuple[Policy, PolicyVersion]]:
    """Every policy with a published version, scoped to ``role_ids``.

    ``role_ids=None`` returns only policies with NO audience rows — the public
    set, which is what an unauthenticated reader gets. A non-empty list
    returns those plus every policy naming one of the given roles.

    Takes role IDS rather than codes because resolving a code to a role is the
    roles catalogue's job, and that lives behind another feature's public API.
    """
    has_audience = (
        select(func.count())
        .select_from(PolicyAudienceRole)
        .where(PolicyAudienceRole.policy_id == Policy.id)
        .correlate(Policy)
        .scalar_subquery()
    )

    stmt = (
        select(Policy, PolicyVersion)
        .join(PolicyVersion, PolicyVersion.policy_id == Policy.id)
        .where(
            PolicyVersion.status == "published",
            PolicyVersion.language == language,
        )
    )

    if role_ids:
        matches_role = (
            select(func.count())
            .select_from(PolicyAudienceRole)
            .where(
                PolicyAudienceRole.policy_id == Policy.id,
                PolicyAudienceRole.role_id.in_(list(role_ids)),
            )
            .correlate(Policy)
            .scalar_subquery()
        )
        stmt = stmt.where((has_audience == 0) | (matches_role > 0))
    else:
        stmt = stmt.where(has_audience == 0)

    stmt = stmt.order_by(Policy.category, Policy.slug)
    return [(p, v) for p, v in (await db.execute(stmt)).all()]


__all__ = [
    "audience_role_ids",
    "get_policy",
    "get_policy_by_slug",
    "get_version",
    "insert_policy",
    "insert_version",
    "latest_version",
    "list_policies",
    "list_versions",
    "open_draft",
    "published_documents",
    "published_version",
    "replace_audience",
    "supersede_published",
]
