from abridgeai.features.materials.workers.cron import cleanup_orphaned_uploads_task
from abridgeai.features.materials.workers.ingest import ingest_material_version_task

JOBS = [ingest_material_version_task]
CRON_TASKS = [cleanup_orphaned_uploads_task]

__all__ = ["CRON_TASKS", "JOBS", "cleanup_orphaned_uploads_task", "ingest_material_version_task"]
