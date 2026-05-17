from abridgeai.features.spaced_repetition.workers.scan_due_cards import scan_due_cards_task

JOBS = [scan_due_cards_task]
CRON_TASKS = [scan_due_cards_task]

__all__ = ["CRON_TASKS", "JOBS", "scan_due_cards_task"]
