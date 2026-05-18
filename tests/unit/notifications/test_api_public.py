from __future__ import annotations

from abridgeai.features.notifications.api.public import (
    EMAIL_NOTIFICATION_TASK_NAME,
    send_notification,
)
from abridgeai.features.notifications.services import dispatch as _dispatch


def test_send_notification_is_true_alias() -> None:
    assert send_notification is _dispatch.send_notification


def test_email_task_name_is_true_alias() -> None:
    assert EMAIL_NOTIFICATION_TASK_NAME is _dispatch.EMAIL_NOTIFICATION_TASK_NAME
    assert EMAIL_NOTIFICATION_TASK_NAME == "send_email_notification_task"


def test_module_exports() -> None:
    from abridgeai.features.notifications.api import public

    assert {"send_notification", "EMAIL_NOTIFICATION_TASK_NAME"} == set(public.__all__)
