"""Grant the manager permission set to the Faculty Dean (hod) role.

Product decision: the Dean is a SUPERSET of Manager — every screen a manager
can open, a dean must open too (plus dean-specific powers like
``user.role_assign.hod`` and ``learning_program.switch.review`` which the
manager does not hold). The reverse must NOT hold: a manager gains nothing
dean-only. Before this migration the ``hod`` role held only its oversight
permissions and none of the manager's course/enrollment/user-admin set, so
dean accounts hit 403s on manager screens.

Migration 0004 already seeded role_permissions on existing databases, so
editing ``role_seeds.yaml`` alone does not reach them — this migration applies
the delta idempotently, following the 0071 pattern.

Downgrade removes exactly the granted rows (the intersection), returning hod
to its pre-migration set.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0085_dean_superset_of_manager"
down_revision = "0084_drop_lock_quiz_ef"
branch_labels = None
depends_on = None

_ROLE_CODE = "hod"

# The full manager set from role_seeds.yaml at this revision. Every code here
# already exists in the permissions table (manager holds it), so no
# permission-row insertion is needed — grants only.
_MANAGER_PERMISSIONS = (
    "bulk_import.read",
    "course.assign_teacher",
    "course.create",
    "course.delete",
    "course.enrollment.create",
    "course.enrollment.read",
    "course.enrollment.remove",
    "course.publish",
    "course.read",
    "course.read.draft",
    "course.update",
    "learning_outcome.manage",
    "learning_program.enroll",
    "learning_program.manage",
    "learning_program.read",
    "notification.send.bulk",
    "org_unit.manage",
    "system.stats.read",
    "user.bulk_import",
    "user.disable",
    "user.read",
    "user.role_assign",
    "user.role_assign.hod",
)


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        text(
            """
            INSERT INTO role_permissions (role_id, permission_id, created_at)
            SELECT r.id, p.id, NOW()
            FROM roles r
            JOIN permissions p ON p.code = ANY(:codes)
            WHERE r.code = :role_code
              AND p.deleted_at IS NULL
            ON CONFLICT (role_id, permission_id) DO NOTHING
            """
        ),
        {"codes": list(_MANAGER_PERMISSIONS), "role_code": _ROLE_CODE},
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        text(
            """
            DELETE FROM role_permissions
            WHERE role_id = (SELECT id FROM roles WHERE code = :role_code)
              AND permission_id IN (
                  SELECT id FROM permissions WHERE code = ANY(:codes)
              )
            """
        ),
        {"codes": list(_MANAGER_PERMISSIONS), "role_code": _ROLE_CODE},
    )
