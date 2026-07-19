"""Identity feature ORM models.

Ports the legacy ``backend/app/models/{user,auth}.py`` declarations onto the
new ``abridgeai.core.db.Base`` and the canonical mixins. The source of truth
for column names, types, and constraints is
``migrations/versions/0001_baseline_schema.py`` — not the legacy ORM.

Mixin policy (Reconciliation §A13 + draft "Soft-delete eligibility"):
* Credential / session entities (``users``, ``auth_identities``, ``auth_sessions``,
  ``mfa_factors``, ``mfa_recovery_codes``, ``mfa_challenges``) stay HARD-delete.
  They get ``UUIDPrimaryKeyMixin`` + ``TimestampMixin`` (or ``CreatedAtMixin``
  for append-only rows). They never receive ``SoftDeleteMixin`` or
  ``AuditedByMixin``.
* Profile + storage entities (``storage_objects``, ``user_profiles``,
  ``user_profile_links``) are listed in ``SOFT_DELETE_TABLES`` in the baseline
  migration. They receive ``AuditedByMixin`` + ``SoftDeleteMixin`` so the audit
  columns the migration added line up with the ORM metadata.

OAuth-only forever: there is intentionally no ``password_hash`` column on
``User``. The decision is locked (plan §16).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import CITEXT, INET
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import Base
from abridgeai.core.db.mixins import (
    PGUUID,
    AuditedByMixin,
    CreatedAtMixin,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Application user (OAuth-only; no ``password_hash`` column)."""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'invited', 'inactive', 'suspended')",
            name="ck_users_status",
        ),
    )

    primary_email: Mapped[str] = mapped_column(CITEXT(), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    last_login_at: Mapped[datetime | None] = mapped_column(nullable=True)


class StorageObject(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    Base,
):
    __tablename__ = "storage_objects"
    __table_args__ = (
        UniqueConstraint("bucket", "object_key", name="uq_storage_objects_bucket_key"),
    )

    bucket: Mapped[str] = mapped_column(String(100), nullable=False)
    object_key: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    uploaded_at: Mapped[datetime] = mapped_column(nullable=False)


class UserProfile(TimestampMixin, AuditedByMixin, SoftDeleteMixin, Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    given_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    family_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    avatar_object_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("storage_objects.id", ondelete="SET NULL"),
        nullable=True,
    )
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Preferred UI + notification language ('en' | 'vi'). NULL = not set;
    # notification dispatch normalizes NULL to 'en' (matches the frontend
    # i18next ``fallbackLng``). CHECK constraint added in migration 0024.
    locale: Mapped[str | None] = mapped_column(String(5), nullable=True)


class UserProfileLink(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    AuditedByMixin,
    SoftDeleteMixin,
    Base,
):
    __tablename__ = "user_profile_links"
    __table_args__ = (
        CheckConstraint(
            "link_type IN ('website', 'github', 'linkedin', 'portfolio', 'other')",
            name="ck_user_profile_links_link_type",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    link_type: Mapped[str] = mapped_column(String(30), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str | None] = mapped_column(String(100), nullable=True)


class AuthIdentity(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "auth_identities"
    __table_args__ = (
        UniqueConstraint("provider", "provider_subject", name="uq_auth_identity_provider_subject"),
        CheckConstraint("provider IN ('google')", name="ck_auth_identities_provider"),
        Index("idx_auth_identities_user", "user_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    provider_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_email: Mapped[str | None] = mapped_column(CITEXT(), nullable=True)


class AuthSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (Index("idx_auth_sessions_user", "user_id"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    refresh_token_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(nullable=True)
    mfa_verified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    ip_address: Mapped[str | None] = mapped_column(INET, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)


class MfaFactor(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "mfa_factors"
    __table_args__ = (
        CheckConstraint("factor_type IN ('totp')", name="ck_mfa_factors_factor_type"),
        Index("idx_mfa_factors_user", "user_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    factor_type: Mapped[str] = mapped_column(String(20), nullable=False)
    secret_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    disabled_at: Mapped[datetime | None] = mapped_column(nullable=True)


class MfaRecoveryCode(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "mfa_recovery_codes"
    __table_args__ = (
        UniqueConstraint("factor_id", "code_hash", name="uq_mfa_recovery_factor_code"),
    )

    factor_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("mfa_factors.id", ondelete="CASCADE"),
        nullable=False,
    )
    code_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(nullable=True)


class MfaChallenge(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Short-lived MFA challenge nonce; ``consumed_at`` marks completion.

    Append-only style — only ``created_at`` (no ``updated_at``) per baseline DDL.
    """

    __tablename__ = "mfa_challenges"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    factor_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("mfa_factors.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("auth_sessions.id", ondelete="CASCADE"),
        nullable=True,
    )
    expires_at: Mapped[datetime] = mapped_column(nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(nullable=True)


__all__ = [
    "AuthIdentity",
    "AuthSession",
    "MfaChallenge",
    "MfaFactor",
    "MfaRecoveryCode",
    "StorageObject",
    "User",
    "UserProfile",
    "UserProfileLink",
]
