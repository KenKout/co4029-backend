from abridgeai.features.notifications.workers.email import send_email_notification_task

JOBS = [send_email_notification_task]

__all__ = ["JOBS", "send_email_notification_task"]
