"""seed permission catalog

Revision ID: 0004_seed_permission_catalog
Revises: 0003_fk_cascade_no_action
Create Date: 2026-05-16 23:00:00.000000

Seeds the access-control catalog from the YAML single source of truth and
inserts the platform's stable ``system`` user.

Source of truth
---------------
``abridgeai/access_control/permissions/catalog.yaml`` and
``abridgeai/access_control/permissions/role_seeds.yaml`` are loaded via
:mod:`abridgeai.access_control.permissions.loader`. The loader cross-validates
that every role-permission code resolves to a catalog entry and expands the
``"ALL"`` sentinel for the admin role into the full catalog code list, so this
migration never has to know either fact.

System user
-----------
A single users row with stable UUID ``00000000-0000-0000-0000-000000000001``
and email ``system@abridgeai.local`` is inserted FIRST. It is the platform's
automation actor referenced by audit FKs when no HTTP/worker context is set.
Status is ``inactive`` (the only enum value compatible with "never logs in,
never receives mail" - the baseline ``users.status`` CHECK constraint allows
only ``active|invited|inactive|suspended``; there is no ``system`` value).

Idempotency
-----------
Every INSERT uses ``ON CONFLICT ... DO NOTHING`` so re-running the migration
on a database that already has the seed rows is a no-op. The system user
conflicts on its primary key, role_permissions on its composite primary key,
and permissions/roles on their PARTIAL unique index ``(code) WHERE deleted_at
IS NULL`` (T0.7 soft-delete pattern - so the conflict target must include the
``WHERE deleted_at IS NULL`` predicate).

Downgrade caveat
----------------
The downgrade DELETEs role_permissions, then roles, then permissions, then the
system user. It is only safe to run BEFORE any user_role_assignments rows
exist that reference these roles (i.e. on a fresh database, immediately after
the upgrade). Once T1.13 (or the application) creates user_role_assignments
pointing at these roles, the role DELETE will fail with a foreign-key
violation. The test suite exercises the round-trip on a clean database; do
not attempt the downgrade on a populated production environment.
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

from abridgeai.access_control.permissions.loader import load_catalog, load_role_seeds

revision = "0004_seed_permission_catalog"
down_revision = "0003_fk_cascade_no_action"
branch_labels = None
depends_on = None


# Stable UUID for the platform automation actor. Hard-coded so audit FKs are
# reproducible across environments and so deployments never re-create the row
# under a fresh UUID.
SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"
SYSTEM_USER_EMAIL = "system@abridgeai.local"


def upgrade() -> None:
    """Insert system user, then permissions, roles, role_permissions."""
    catalog = load_catalog()
    role_seeds = load_role_seeds(catalog)

    bind = op.get_bind()

    # 1. System user FIRST: audit FKs on permissions/roles/role_permissions
    #    use SET NULL on user delete, but inserting the parent row first makes
    #    future audit-listener seeding (where ``created_by`` could point here)
    #    a non-event. status='inactive' means "never receives login or mail".
    bind.execute(
        text(
            """
            INSERT INTO users (id, primary_email, status, created_at, updated_at)
            VALUES (
                CAST(:user_id AS uuid),
                :email,
                'inactive',
                NOW(),
                NOW()
            )
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {"user_id": SYSTEM_USER_ID, "email": SYSTEM_USER_EMAIL},
    )

    # 2. Permissions: one row per catalog entry. ``name`` mirrors ``code``
    #    because the YAML only carries (code, description, category); the
    #    permissions table has no category column, so the category is
    #    discarded at the DB layer (still available via the YAML for admin UIs).
    for perm in catalog.permissions:
        bind.execute(
            text(
                """
                INSERT INTO permissions (id, code, name, description, created_at, updated_at)
                VALUES (
                    uuid_generate_v4(),
                    :code,
                    :name,
                    :description,
                    NOW(),
                    NOW()
                )
                ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING
                """
            ),
            {
                "code": perm.code,
                "name": perm.code,
                "description": perm.description,
            },
        )

    # 3. Roles: the canonical 5. ``is_system_role`` defaults to TRUE in the
    #    baseline DDL; we set it explicitly so the seed origin is unambiguous.
    #    The RoleDef ``description`` is YAML-only metadata - the roles table
    #    has no description column, so it is discarded at the DB layer.
    for role in role_seeds.roles:
        bind.execute(
            text(
                """
                INSERT INTO roles (id, code, name, is_system_role, created_at, updated_at)
                VALUES (
                    uuid_generate_v4(),
                    :code,
                    :name,
                    TRUE,
                    NOW(),
                    NOW()
                )
                ON CONFLICT (code) WHERE deleted_at IS NULL DO NOTHING
                """
            ),
            {"code": role.code, "name": role.name},
        )

    # 4. role_permissions: subquery joins on code so we resolve to the UUIDs
    #    that ``uuid_generate_v4()`` just minted. ``role.permissions`` is
    #    always a concrete list at this point because the loader expanded the
    #    "ALL" sentinel for the admin role.
    for role in role_seeds.roles:
        permissions_codes = role.permissions
        # Type guard: the loader always returns a list after expansion.
        if not isinstance(permissions_codes, list):  # pragma: no cover - defensive
            msg = f"loader returned unexpanded sentinel for role {role.code!r}"
            raise RuntimeError(msg)
        for perm_code in permissions_codes:
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
                {"role_code": role.code, "perm_code": perm_code},
            )


def downgrade() -> None:
    """Reverse of upgrade: peel rows back in FK-safe order.

    Read the YAML again so the deletion set tracks the catalog precisely; this
    means a downgrade after a YAML edit removes only the rows the YAML still
    knows about. Pre-existing rows (from a hypothetical earlier seed) survive.
    """
    catalog = load_catalog()
    role_seeds = load_role_seeds(catalog)

    permission_codes = sorted(catalog.codes())
    role_codes = [r.code for r in role_seeds.roles]

    bind = op.get_bind()

    # 1. role_permissions where BOTH the role and the permission belong to
    #    this seed. A user_role_assignments row pointing at a seeded role
    #    would NOT block this DELETE (role_permissions is a different table),
    #    but the role DELETE below would.
    bind.execute(
        text(
            """
            DELETE FROM role_permissions
            WHERE role_id IN (
                SELECT id FROM roles WHERE code = ANY(:role_codes)
            )
            AND permission_id IN (
                SELECT id FROM permissions WHERE code = ANY(:permission_codes)
            )
            """
        ),
        {"role_codes": role_codes, "permission_codes": permission_codes},
    )

    # 2. Roles. Will fail if any user_role_assignments row references a
    #    seeded role - intended (downgrade is unsafe post-T1.13, see docstring).
    bind.execute(
        text("DELETE FROM roles WHERE code = ANY(:codes)"),
        {"codes": role_codes},
    )

    # 3. Permissions. Will fail if any user_permission_grants row references
    #    a seeded permission.
    bind.execute(
        text("DELETE FROM permissions WHERE code = ANY(:codes)"),
        {"codes": permission_codes},
    )

    # 4. System user. SET NULL FKs from audit columns make this safe even if
    #    other rows referenced it.
    bind.execute(
        text("DELETE FROM users WHERE id = CAST(:user_id AS uuid)"),
        {"user_id": SYSTEM_USER_ID},
    )
