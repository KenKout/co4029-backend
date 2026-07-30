"""Look for rows written across an organization boundary.

Several routes shipped without an organization check (see the commit
"Enforce organization scope on every org-owned route"). The worst was
``POST /admin/organizations/{org_id}/memberships``, which let anyone holding
``org_unit.manage`` in any tenant write a membership into any other — and a
membership is what every downstream org check reads, so a single such row
legitimises everything that follows it.

The routes are closed now. This answers the separate question: was any of it
used before they were? Run it per environment.

    uv run --no-sync python scripts/audit_org_tenancy.py

What "suspicious" means here
----------------------------
A row whose ``created_by`` / ``updated_by`` user has no active membership in
the organization that owns the row. That is a heuristic, not a verdict, and it
over-reports in three known ways:

* **Seed and migration accounts.** Rows written by a system account are
  reported and grouped by actor so they are easy to recognise and dismiss.
* **Platform administrators.** ``system.administer`` is granted globally and
  is *meant* to act across tenants; holders are labelled as such below.
* **Membership at the time vs now.** Only current membership is checked, so an
  actor who has since left the organization looks like an intruder, and the
  first membership in a brand-new organization is necessarily written by
  someone who is not yet a member.

So read the output as a shortlist to eyeball, not a list of incidents. What
matters is whether any actor appears that is neither a system account nor a
platform admin.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy import text

from abridgeai.core.db import get_sessionmaker

# (table, actor column). Every table here is org-owned and course-less — the
# family whose routes lacked a check.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("organization_memberships", "created_by"),
    ("organization_memberships", "updated_by"),
    ("career_paths", "updated_by"),
    ("course_invitation_codes", "updated_by"),
    ("courses", "updated_by"),
    ("org_units", "updated_by"),
    ("organization_domains", "updated_by"),
)

_SUSPICIOUS_SQL = """
    SELECT t.{actor} AS actor_id,
           u.primary_email AS actor_email,
           u.status AS actor_status,
           count(*) AS rows_touched
    FROM {table} t
    JOIN users u ON u.id = t.{actor}
    WHERE t.{actor} IS NOT NULL
      AND t.deleted_at IS NULL
      AND NOT EXISTS (
          SELECT 1 FROM organization_memberships m
          WHERE m.user_id = t.{actor}
            AND m.organization_id = t.organization_id
            AND m.status = 'active'
            AND m.deleted_at IS NULL
      )
    GROUP BY 1, 2, 3
    ORDER BY count(*) DESC
"""

_IS_PLATFORM_ADMIN_SQL = text(
    """
    SELECT 1
    FROM permissions p
    JOIN role_permissions rp ON rp.permission_id = p.id
    JOIN user_role_assignments ura ON ura.role_id = rp.role_id
    WHERE ura.user_id = :user_id
      AND p.code = 'system.administer'
      AND ura.deleted_at IS NULL
    LIMIT 1
    """
)


@dataclass
class Finding:
    table: str
    actor_column: str
    actor_id: str
    actor_email: str
    actor_status: str
    rows_touched: int
    is_platform_admin: bool

    @property
    def benign(self) -> bool:
        """True when the actor is expected to write across tenants."""
        return self.is_platform_admin or self.actor_email.endswith("@abridgeai.local")


async def main() -> int:
    sessionmaker = get_sessionmaker()
    findings: list[Finding] = []
    admin_cache: dict[str, bool] = {}

    async with sessionmaker() as session:
        for table, actor in _TARGETS:
            try:
                rows = (
                    await session.execute(
                        text(_SUSPICIOUS_SQL.format(table=table, actor=actor))
                    )
                ).mappings().all()
            except Exception as exc:  # noqa: BLE001 -- a missing column is data, not a crash
                await session.rollback()
                print(f"  skipped {table}.{actor}: {str(exc).splitlines()[0]}")
                continue

            for row in rows:
                actor_id = str(row["actor_id"])
                if actor_id not in admin_cache:
                    hit = (
                        await session.execute(_IS_PLATFORM_ADMIN_SQL, {"user_id": actor_id})
                    ).first()
                    admin_cache[actor_id] = hit is not None
                findings.append(
                    Finding(
                        table=table,
                        actor_column=actor,
                        actor_id=actor_id,
                        actor_email=row["actor_email"] or "",
                        actor_status=row["actor_status"] or "",
                        rows_touched=int(row["rows_touched"]),
                        is_platform_admin=admin_cache[actor_id],
                    )
                )

    if not findings:
        print("No rows written by a non-member. Nothing to review.")
        return 0

    needs_review = [f for f in findings if not f.benign]
    benign = [f for f in findings if f.benign]

    if needs_review:
        print("REVIEW — actor is neither a platform admin nor a system account:\n")
        for f in needs_review:
            print(
                f"  {f.table}.{f.actor_column}: {f.rows_touched} row(s) by "
                f"{f.actor_email} ({f.actor_id}, status={f.actor_status})"
            )
        print()
    else:
        print("No actor requires review.\n")

    if benign:
        print("Expected cross-tenant writers (platform admins / system accounts):\n")
        for f in benign:
            label = "platform admin" if f.is_platform_admin else "system account"
            print(
                f"  {f.table}.{f.actor_column}: {f.rows_touched} row(s) by "
                f"{f.actor_email} [{label}]"
            )

    # Exit 1 when something needs a human, so this can gate a pipeline.
    return 1 if needs_review else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
