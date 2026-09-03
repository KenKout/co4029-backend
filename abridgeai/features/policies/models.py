"""Policies feature ORM models.

Platform policy documents — privacy, terms, and the academic-operational set
(course, quiz, interview, …) — as a first-class entity rather than hardcoded
front-end text.

Three tables (migration 0102):

* ``policies`` — identity only: slug, category. Renaming the document never
  breaks a link, because the slug lives here and every version hangs off this
  id rather than off a title.
* ``policy_versions`` — a frozen revision carrying the body, the language, the
  publisher and the publication date.
* ``policy_audience_roles`` — which roles the policy names as a party.

The identity/version split is deliberately the same shape as
``career_paths`` / ``career_path_versions`` (migration 0074): editing stays
cheap on the identity row, while a published version freezes exactly what a
reader was shown. The immutability rule is the same too — a published version
is never edited in place; a new draft is opened and published in its turn.

AUDIENCE IS A JOIN TABLE, NOT AN ARRAY COLUMN. Nothing else in this codebase
uses a Postgres ARRAY, and a join keeps the set queryable and referentially
honest against the roles catalogue — so the label an admin picks is the role's
real name, never a second copy of it typed into prose. **No rows at all means
public**: privacy and terms bind every reader and must stay readable before an
account exists.

Mixin policy mirrors the content aggregate (quizzes / lessons / discussions):
all three tables are admin-editable content, so they carry
``UUIDPrimaryKeyMixin`` + ``TimestampMixin`` + ``AuditedByMixin`` +
``SoftDeleteMixin``. Soft-delete is enforced globally by the loader criteria
listener (``core.db.soft_delete``).

Uniqueness follows the 0002 house rule: every unique key on a soft-deletable
table is a PARTIAL index filtered on ``deleted_at IS NULL``, so archiving a
policy frees its slug for reuse instead of reserving it forever.

FK on-delete policy (see ``scripts/audit_fks.py``): parents carrying
``SoftDeleteMixin`` are soft-deleted via UPDATE, never hard-deleted on the
happy path, so FKs into them use ``NO ACTION``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from abridgeai.core.db import (
    PGUUID,
    AuditedByMixin,
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

#: Genres of policy document. ``legal`` is the platform-wide set (privacy,
#: terms, cookies) written in sentence-case headings and binding on everyone;
#: ``academic`` is the operational set written on the thirteen-clause spine
#: the Learning Program and Career Path documents established.
POLICY_CATEGORIES = ("legal", "academic")

#: A version is ``draft`` while it is being written, ``published`` once an
#: admin releases it (frozen from then on), and ``archived`` when a newer
#: version supersedes it. The current document for a (policy, language) is the
#: highest ``version_no`` whose status is ``published``.
POLICY_VERSION_STATUSES = ("draft", "published", "archived")


class Policy(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """A policy document's identity. The content lives in its versions."""

    __tablename__ = "policies"
    __table_args__ = (
        CheckConstraint(
            "category IN ('legal', 'academic')",
            name="ck_policies_category",
        ),
        # Partial, per the 0002 rule — a soft-deleted policy must not hold its
        # slug hostage.
        Index(
            "uq_policies_slug",
            "slug",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    slug: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(20), nullable=False)

    versions: Mapped[list[PolicyVersion]] = relationship(
        back_populates="policy",
        cascade="all, delete-orphan",
    )
    audience: Mapped[list[PolicyAudienceRole]] = relationship(
        back_populates="policy",
        cascade="all, delete-orphan",
    )


class PolicyVersion(UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    """A frozen revision of one policy, in one language.

    ``version_no`` is monotonically increasing per (policy, language) — the
    two languages of one document advance independently, so an English
    correction does not silently invalidate a translation nobody has
    re-reviewed.

    ``published_by`` / ``published_at`` are stamped by the publish action
    itself rather than typed into the body, so the attribution can never drift
    from whoever actually released it.
    """

    __tablename__ = "policy_versions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'published', 'archived')",
            name="ck_policy_versions_status",
        ),
        CheckConstraint("version_no > 0", name="ck_policy_versions_version_no"),
        # A published version must carry its date; anything else must not
        # pretend to have one. Enforced in the schema rather than only in the
        # service so a direct SQL fix cannot produce a policy that claims to
        # have been published on no date.
        CheckConstraint(
            "(status = 'published') = (published_at IS NOT NULL)",
            name="ck_policy_versions_published_at",
        ),
        Index(
            "uq_policy_versions_policy_language_version",
            "policy_id",
            "language",
            "version_no",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    policy_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    #: BCP-47. English-only today; a second language is another row per
    #: (policy_id, version_no), not a schema change.
    language: Mapped[str] = mapped_column(String(10), nullable=False, server_default=text("'en'"))
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'draft'"))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Markdown, nh3-sanitized on every write by the service layer.
    body: Mapped[str] = mapped_column(Text, nullable=False)
    #: Format discriminator matching the quiz rich-content convention, so the
    #: same ``RichContent`` renderer handles both. Only ``markdown`` today.
    #:
    #: Both defaults are deliberate: the server default covers rows written by
    #: SQL (migrations, seeds), and the Python default means a freshly flushed
    #: instance already carries the value instead of depending on the INSERT
    #: fetching it back — the read schema requires a string, not ``None``.
    format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="markdown",
        server_default=text("'markdown'"),
    )
    #: What changed since the previous version, shown to a reader who has
    #: already seen an older one. Optional — a first version has no "since".
    changelog: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
    )

    policy: Mapped[Policy] = relationship(back_populates="versions")


class PolicyAudienceRole(
    UUIDPrimaryKeyMixin, TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base
):
    """One role this policy names as a party.

    A role sees the policy in its index exactly when a row exists here. No
    rows at all means public — the convention privacy/terms/cookies already
    rely on, and the reason audience is absence-based rather than a flag.

    Points at ``roles.id``, NOT ``roles.code``: migration 0002 replaced the
    full unique constraint on ``code`` with a PARTIAL unique index
    (``WHERE deleted_at IS NULL``), and PostgreSQL cannot back a foreign key
    with a partial index. ``user_role_assignments`` resolves the same way, so
    reading a role's code or display name is the same one-hop join the rest of
    the codebase already does.
    """

    __tablename__ = "policy_audience_roles"
    __table_args__ = (
        Index(
            "uq_policy_audience_roles_policy_role",
            "policy_id",
            "role_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    policy_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("policies.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="NO ACTION"),
        nullable=False,
        index=True,
    )

    policy: Mapped[Policy] = relationship(back_populates="audience")


__all__ = [
    "POLICY_CATEGORIES",
    "POLICY_VERSION_STATUSES",
    "Policy",
    "PolicyAudienceRole",
    "PolicyVersion",
]
