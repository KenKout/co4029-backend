"""Backfill action_url for legacy notifications created before 0029.

Revision ID: 0030_backfill_action_url
Revises: 0029_notification_action_url
Create Date: 2026-07-21 00:00:00.000000

Only touches rows where ``action_url IS NULL`` (idempotent -- re-running is a
no-op). Two legacy notification shapes exist, both spaced_repetition:

* ``entity_type = 'quiz_question'`` -> resolve question -> quiz -> course slug
  and point at the learner "continue learning" route
  (``/courses/{slug}/learn``), matching what the remediation dispatcher now
  writes for new rows. Rows whose question/quiz/course can no longer be
  resolved (deleted content) are left NULL -- still informational-only, no
  dead link.
* ``entity_type IS NULL`` (cross-course due-cards summaries) -> ``/progress``,
  matching the scan-due-cards worker.

Downgrade is intentionally a no-op: a backfill can't be precisely reversed
(we can't distinguish rows we filled from rows later set by normal dispatch),
and NULLing by value would clobber legitimately-set new notifications.

NOTE: revision id kept short -- ``alembic_version.version_num`` is
``varchar(32)``, so ids must stay <= 32 chars.
"""

from __future__ import annotations

from alembic import op

revision = "0030_backfill_action_url"
down_revision = "0029_notification_action_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # quiz_question SR notifications -> /courses/{slug}/learn, resolved via the
    # same question->quiz->course join the remediation dispatcher uses.
    op.execute(
        """
        UPDATE notifications AS n
        SET action_url = '/courses/' || c.slug || '/learn'
        FROM quiz_questions qq
        JOIN quizzes q ON q.id = qq.quiz_id
        JOIN courses c ON c.id = q.course_id
        WHERE n.action_url IS NULL
          AND n.entity_type = 'quiz_question'
          AND n.entity_id = qq.id
        """
    )

    # Cross-course due-cards summaries (no entity) -> learner progress page.
    op.execute(
        """
        UPDATE notifications
        SET action_url = '/progress'
        WHERE action_url IS NULL
          AND category = 'spaced_repetition'
          AND entity_type IS NULL
        """
    )


def downgrade() -> None:
    # No-op: backfilled data cannot be precisely distinguished from rows set
    # by normal dispatch after this migration ran, so we do not attempt to
    # reverse it (see module docstring).
    pass
