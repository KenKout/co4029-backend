from abridgeai.features.notifications.queries.notifications import (
    bulk_mark_read,
    cursor_list_my_notifications,
    dismiss_notification,
    get_my_notification,
    get_my_unread_count,
    insert_notification,
    mark_as_read,
)
from abridgeai.features.notifications.queries.preferences import (
    get_email_preference,
    list_my_preferences,
    upsert_preference,
)

__all__ = [
    "bulk_mark_read",
    "cursor_list_my_notifications",
    "dismiss_notification",
    "get_email_preference",
    "get_my_notification",
    "get_my_unread_count",
    "insert_notification",
    "list_my_preferences",
    "mark_as_read",
    "upsert_preference",
]
