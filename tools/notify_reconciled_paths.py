"""Tell students whose duplicate career path was closed by a reconciliation.

Migration ``0131`` closed duplicate active attempts — the same career path
running in two of one student's programs, which the database now forbids — and
recorded why on each attempt's ``exit_snapshot``. It could not notify anyone:
an Alembic migration runs on a sync connection and has no access to the
notification service.

This is that half. A student whose record changed without them touching it is
owed an explanation, and the information to write one is already on the row.

Idempotent by design: each notified attempt gets
``reconciliation_notified_at`` stamped into the same ``exit_snapshot``, and
rows carrying it are skipped. Re-running is safe, and running it in a database
where the reconciliation found nothing does nothing.

Usage::

    uv run python -m tools.notify_reconciled_paths --dry-run
    uv run python -m tools.notify_reconciled_paths

``--dry-run`` prints what would be sent and writes nothing, which is the way
to check the copy against real rows before any student sees it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from abridgeai.core.db import get_sessionmaker
from abridgeai.features.identity.api import public as identity_api
from abridgeai.features.notifications.api import public as notifications_api

_RECONCILED_BY = "0131_student_scoped_active_path"

# ``system`` rather than ``path_change_review``: no Faculty Dean reviewed
# this and no request was decided. It is a platform correction, and saying so
# keeps the review category meaning what it says.
_CATEGORY = "system"

# Everything the message needs, resolved in one statement. ``kept`` is joined
# through the attempt the reconciliation preserved, which is why the closed
# row stored its id rather than only a reason.
_PENDING_SQL = """
SELECT
    closed.id                AS attempt_id,
    closed.student_id        AS student_id,
    path.name                AS path_name,
    closed_program.name      AS closed_program_name,
    kept_program.name        AS kept_program_name
FROM program_path_attempts AS closed
JOIN career_paths AS path
  ON path.id = closed.career_path_id
JOIN program_enrollments AS closed_enrollment
  ON closed_enrollment.id = closed.program_enrollment_id
JOIN learning_programs AS closed_program
  ON closed_program.id = closed_enrollment.learning_program_id
JOIN program_path_attempts AS kept
  ON kept.id = (closed.exit_snapshot ->> 'kept_attempt_id')::uuid
JOIN program_enrollments AS kept_enrollment
  ON kept_enrollment.id = kept.program_enrollment_id
JOIN learning_programs AS kept_program
  ON kept_program.id = kept_enrollment.learning_program_id
WHERE closed.exit_snapshot ->> 'reconciled_by' = :marker
  AND closed.exit_snapshot ->> 'reconciliation_notified_at' IS NULL
ORDER BY closed.ended_at
"""

_STAMP_SQL = """
UPDATE program_path_attempts
SET exit_snapshot = exit_snapshot || jsonb_build_object(
        'reconciliation_notified_at', :stamped_at
    )
WHERE id = :attempt_id
"""


async def _notify_all(*, dry_run: bool) -> int:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        rows: list[Any] = (
            await db.execute(text(_PENDING_SQL), {"marker": _RECONCILED_BY})
        ).mappings().all()

        if not rows:
            print("nothing to notify: no unnotified reconciliations in this database")
            return 0

        for row in rows:
            locale = await identity_api.get_user_locale(db, row["student_id"])
            title = notifications_api.path_reconciled_title(
                path_name=row["path_name"], locale=locale
            )
            body = notifications_api.path_reconciled_body(
                path_name=row["path_name"],
                kept_program_name=row["kept_program_name"],
                closed_program_name=row["closed_program_name"],
                locale=locale,
            )

            if dry_run:
                print(f"[dry-run] student {row['student_id']} ({locale}): {title}")
                print(f"          {body}")
                continue

            await notifications_api.send_notification(
                db,
                recipient_user_id=row["student_id"],
                notification_type=_CATEGORY,
                title=title,
                body=body,
                entity_type="program_path_attempt",
                entity_id=row["attempt_id"],
                action_url="/me/learning-programs",
                # No arq pool: in-app only. This explains a correction that
                # has already happened and costs the student nothing to read
                # late, so it does not warrant an email.
                arq_pool=None,
            )
            await db.execute(
                text(_STAMP_SQL),
                {
                    "attempt_id": row["attempt_id"],
                    "stamped_at": datetime.now(tz=UTC).isoformat(),
                },
            )
            print(f"notified student {row['student_id']} about {row['path_name']}")

        if dry_run:
            print(f"[dry-run] {len(rows)} notification(s) would be sent; nothing written")
            return 0

        # One transaction: a notification whose stamp did not land would be
        # sent again on the next run.
        await db.commit()
        print(f"sent {len(rows)} notification(s)")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be sent and write nothing",
    )
    args = parser.parse_args()
    return asyncio.run(_notify_all(dry_run=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
