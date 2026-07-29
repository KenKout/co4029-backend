"""Add the ``learning_outcome.manage`` permission and grant it to manager + admin.

Learning-outcome authoring split (manager-owned LOs): LO create/edit/delete
was gated on ``course.update``, which teachers also hold — so teachers could
author LOs. The intended model is that a course's learning outcomes are owned
by the **manager** (student + course management), while teachers manage course
*content*. This migration introduces a dedicated ``learning_outcome.manage``
permission and grants it to the ``manager`` and ``admin`` roles only.

Migration 0004 already seeded the catalog on existing databases, so editing
``catalog.yaml`` / ``role_seeds.yaml`` alone does not reach them — this
migration applies the delta idempotently.

The endpoint gates (require_course_permission with allow_owner=False) enforce
this permission and deliberately bypass the course-owner short-circuit, so a
teacher who owns a course still cannot author its LOs.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0067_lo_manage_perm"
down_revision = "0066_org_settings"
branch_labels = None
depends_on = None

_PERMISSION_CODE = "learning_outcome.manage"
_PERMISSION_DESCRIPTION = "Create, edit, or delete a course's learning outcomes (manager-owned)"
# admin holds ALL permissions in role_seeds; grant explicitly here too so an
# existing admin role (seeded before this permission existed) picks it up.
_ROLES_TO_GRANT = ("manager", "admin")


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Permission row (idempotent: partial unique index on code where not deleted).
    bind.execute(
        text(
            """
            INSERT INTO permissions (id, code, name, description, created_at, updated_at)
            VALUES (uuid_generate_v4(), :code, :code, :description, NOW(), NOW())
            ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING
            """
        ),
        {"code": _PERMISSION_CODE, "description": _PERMISSION_DESCRIPTION},
    )

    # 2. Grant to manager + admin (idempotent on the composite PK).
    for role_code in _ROLES_TO_GRANT:
        bind.execute(
            text(
                """
                INSERT INTO role_permissions (role_id, permission_id, created_at)
                SELECT r.id, p.id, NOW()
                FROM roles r
                JOIN permissions p ON p.code = :perm_code
                WHERE r.code = :role_code
                ON CONFLICT (role_id, permission_id) DO NOTHING
                """
            ),
            {"role_code": role_code, "perm_code": _PERMISSION_CODE},
        )


def downgrade() -> None:
    bind = op.get_bind()

    # Remove the grants first (FK), then the permission itself.
    bind.execute(
        text(
            """
            DELETE FROM role_permissions
            WHERE permission_id IN (
                SELECT id FROM permissions WHERE code = :code
            )
            """
        ),
        {"code": _PERMISSION_CODE},
    )
    bind.execute(
        text("DELETE FROM permissions WHERE code = :code"),
        {"code": _PERMISSION_CODE},
    )
