"""Add the ``user.disable`` permission and grant it to manager + admin.

Org-scoped account administration (manager user screen): managers see the
users of their own organization (managers, teachers, students) and can
disable / re-enable accounts that belong to their org — but never other
managers (peer protection). This migration introduces a dedicated
``user.disable`` permission and grants it to the ``manager`` and ``admin``
roles only; the endpoint applies the org-scope + peer guards.

Migration 0004 already seeded the catalog on existing databases, so editing
``catalog.yaml`` / ``role_seeds.yaml`` alone does not reach them — this
migration applies the delta idempotently.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "0071_user_disable_perm"
down_revision = "0070_career_path_stages"
branch_labels = None
depends_on = None

_PERMISSION_CODE = "user.disable"
_PERMISSION_DESCRIPTION = "Disable or re-enable user accounts within scope (org-scoped)"
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
