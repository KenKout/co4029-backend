"""Notification + NotificationPreference ORM models (T7.4).

Schema follows the baseline DDL in ``migrations/versions/0001_baseline_schema.py``
verbatim (Reconciliation §B5). Differences from the task-body draft (which
were resolved against §B5):

* Recipient column is named ``user_id`` (not ``recipient_user_id``).
* Notification taxonomy is ``category`` (not ``type``); CHECK constraint
  enumerates ``{spaced_repetition, lesson_unlock, interview_result,
  course_announcement, system}``.
* Linking columns are ``entity_type`` + ``entity_id`` (not ``link_url``).
* No ``payload`` JSONB column.
* No ``SoftDeleteMixin`` -- §B5 explicitly forbids it. Dismissal is modelled
  as ``delivery_status='cancelled'`` (matches the baseline CHECK constraint
  values: ``pending``, ``sent``, ``failed``, ``cancelled``).
* Audit columns: only ``updated_by`` (added in 0001 outside the soft-delete
  loop). No ``created_by``, no ``deleted_by``. Therefore we hand-roll the
  column instead of mixing in ``AuditedByMixin`` (which would also pull in
  ``created_by``).

``NotificationPreference`` carries the wider 7-value category enum
(``course_updates`` and ``ai_recommendations`` are valid preference rows
even though no notification can be created with those categories today --
the prefs table is forward-compatible).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from abridgeai.core.db import (
    PGUUID,
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

NOTIFICATION_CATEGORIES = (
    "spaced_repetition",
    "lesson_unlock",
    "interview_result",
    "course_announcement",
    "system",
    # Teacher-facing async-job outcomes (each covers success + failure).
    "material_processing",
    "quiz_generation",
    "interview_generation",
)
"""Categories accepted by the ``notifications`` CHECK constraint."""

NOTIFICATION_PREFERENCE_CATEGORIES = (
    "course_updates",
    "ai_recommendations",
    *NOTIFICATION_CATEGORIES,
)
"""Categories accepted by ``notification_preferences`` CHECK constraint."""

NOTIFICATION_CHANNELS = ("email", "in_app")
"""Supported delivery channels."""

NOTIFICATION_DELIVERY_STATUSES = ("pending", "sent", "failed", "cancelled")
"""``delivery_status`` values; ``cancelled`` doubles as user-dismiss."""


class Notification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "notifications"
    __table_args__ = (
        CheckConstraint(
            "category IN ('spaced_repetition', 'lesson_unlock', "
            "'interview_result', 'course_announcement', 'system', "
            "'material_processing', 'quiz_generation', 'interview_generation')",
            name="ck_notifications_category",
        ),
        CheckConstraint(
            "delivery_status IN ('pending', 'sent', 'failed', 'cancelled')",
            name="ck_notifications_delivery_status",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(30))
    entity_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Precomputed relative deep-link (e.g. '/courses/{slug}/quiz/{id}') built
    # by the producing feature at creation time. Frontend renders it as a
    # "Take action" link; NULL means the notification is informational only.
    action_url: Mapped[str | None] = mapped_column(Text)
    scheduled_for: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    # §B5: only updated_by; no created_by, no deleted_by.
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )


class NotificationPreference(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "notification_preferences"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "category",
            "channel",
            name="uq_notification_preferences_user_category_channel",
        ),
        CheckConstraint(
            "category IN ('course_updates', 'ai_recommendations', "
            "'spaced_repetition', 'lesson_unlock', 'interview_result', "
            "'course_announcement', 'system', "
            "'material_processing', 'quiz_generation', 'interview_generation')",
            name="ck_notification_preferences_category",
        ),
        CheckConstraint(
            "channel IN ('email', 'in_app')",
            name="ck_notification_preferences_channel",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("TRUE"))


__all__ = [
    "NOTIFICATION_CATEGORIES",
    "NOTIFICATION_CHANNELS",
    "NOTIFICATION_DELIVERY_STATUSES",
    "NOTIFICATION_PREFERENCE_CATEGORIES",
    "Notification",
    "NotificationPreference",
]
