"""Public, typed cross-feature dispatch API for the notifications feature.

This module re-exports :func:`send_notification` and the related
:data:`EMAIL_NOTIFICATION_TASK_NAME` constant from
:mod:`abridgeai.features.notifications.services.dispatch` so cross-
feature producers (SR remediation + scan-due-cards today; quizzes,
interviews, enrollments tomorrow) can import them via the blessed
``api.public`` path. The current ``ignore_imports`` exceptions for
``spaced_repetition.services.remediation -> notifications.services.dispatch``
and ``spaced_repetition.workers.scan_due_cards -> notifications.services.dispatch``
collapse once those call sites are migrated to this surface in Wave 5.

The plan body referenced ``dispatch_remediation_for_card_failure`` --
the actual callable is the more general :func:`send_notification`,
which the SR card-failure path already invokes with
``notification_type='spaced_repetition'``.

No new logic lives here; this is a typed pass-through. Tests cover the
re-export identity + that the public name resolves to the same callable
as the underlying implementation.
"""

from __future__ import annotations

from abridgeai.features.notifications.services.dispatch import (
    EMAIL_NOTIFICATION_TASK_NAME,
    send_notification,
)

__all__ = [
    "EMAIL_NOTIFICATION_TASK_NAME",
    "send_notification",
]
