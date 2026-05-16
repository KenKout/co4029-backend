"""FK constraint audit (T0.13) — read-only.

Queries PostgreSQL information_schema for every FK in the public schema and
emits a Markdown report flagging CASCADEs whose parent table is on the
soft-delete inventory. Output: ``docs/fk-audit-pre-cascade-fix.md`` at repo
root (one level above ``backend-new/``).

Acceptance: invokable as ``cd backend-new && uv run python scripts/audit_fks.py``.

T0.14 will consume the report to author a CASCADE -> NO ACTION migration.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import psycopg

# ---------------------------------------------------------------------------
# Soft-delete inventory (mirrors migrations/versions/0001_baseline_schema.py
# SOFT_DELETE_TABLES + Reconciliation §A1/§B/§D5). A FK whose referenced
# (parent) table sits in this set must NOT be ``ON DELETE CASCADE``: parents
# are soft-deleted via UPDATE, never hard-deleted on the happy path, so a real
# DELETE on a parent (admin force-purge, GDPR) would silently wipe children.
# Mirrors migrations/versions/0001_baseline_schema.py SOFT_DELETE_TABLES plus
# Reconciliation §A1/§B/§D5 entities. CASCADE FKs into any of these tables
# are unsafe: parents are soft-deleted via UPDATE on the happy path, but a
# real DELETE (admin force-purge, GDPR) would silently wipe the children.
SOFT_DELETABLE_TABLES: frozenset[str] = frozenset(
    {
        "courses",
        "modules",
        "lessons",
        "module_items",
        "lesson_resources",
        "course_learning_outcomes",
        "tags",
        "course_tags",
        "quizzes",
        "quiz_questions",
        "quiz_question_options",
        "quiz_source_lessons",
        "quiz_question_revisions",
        "interview_configs",
        "interview_questions",
        "interview_outcomes",
        "interview_session_questions",
        "interview_session_messages",
        "learning_materials",
        "learning_material_versions",
        "organizations",
        "org_units",
        "organization_memberships",
        "organization_domains",
        "career_paths",
        "career_course_items",
        "student_career_enrollments",
        "permissions",
        "roles",
        "user_role_assignments",
        "user_permission_grants",
        "user_profiles",
        "user_profile_links",
        "storage_objects",
    }
)


FK_QUERY = """
SELECT
    tc.table_name        AS from_table,
    kcu.column_name      AS from_column,
    ccu.table_name       AS to_table,
    ccu.column_name      AS to_column,
    rc.delete_rule       AS delete_rule,
    rc.update_rule       AS update_rule,
    tc.constraint_name   AS constraint_name
FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        AND tc.table_schema = kcu.table_schema
    JOIN information_schema.referential_constraints AS rc
        ON tc.constraint_name = rc.constraint_name
        AND tc.table_schema = rc.constraint_schema
    JOIN information_schema.constraint_column_usage AS ccu
        ON ccu.constraint_name = rc.unique_constraint_name
        AND ccu.constraint_schema = rc.unique_constraint_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;
"""


@dataclass(frozen=True)
class FK:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    delete_rule: str
    update_rule: str
    constraint_name: str

    @property
    def parent_soft_deletable(self) -> bool:
        return self.to_table in SOFT_DELETABLE_TABLES


def _database_url() -> str:
    """Return a psycopg-compatible URL.

    .env stores ``postgresql+psycopg://...`` (SQLAlchemy driver form). Strip
    the ``+psycopg`` so plain ``psycopg.connect()`` accepts it.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        # Fallback to docker-compose default (port 5433 per T0.9)
        url = "postgresql://abridgeai:abridgeai@localhost:5433/abridgeai"
    return url.replace("+psycopg_async", "").replace("+psycopg", "")


def fetch_fks() -> list[FK]:
    with psycopg.connect(_database_url()) as conn, conn.cursor() as cur:
        cur.execute(FK_QUERY)
        rows = cur.fetchall()
    return [FK(*row) for row in rows]


def _md_row(fk: FK) -> str:
    return (
        f"| `{fk.from_table}.{fk.from_column}` "
        f"| `{fk.to_table}` "
        f"| `{fk.delete_rule}` "
        f"| `{fk.update_rule}` "
        f"| {'YES' if fk.parent_soft_deletable else 'no'} "
        f"| `{fk.constraint_name}` |"
    )


def render_report(fks: list[FK]) -> str:
    cascade_to_soft = sorted(
        (fk for fk in fks if fk.delete_rule == "CASCADE" and fk.parent_soft_deletable),
        key=lambda fk: (fk.from_table, fk.from_column),
    )
    other = sorted(
        (
            fk
            for fk in fks
            if not (fk.delete_rule == "CASCADE" and fk.parent_soft_deletable)
        ),
        key=lambda fk: (fk.from_table, fk.from_column),
    )

    rule_counts = Counter(fk.delete_rule for fk in fks)
    rule_summary = ", ".join(
        f"`{rule}`={n}" for rule, n in sorted(rule_counts.items())
    )

    lines: list[str] = []
    lines.append("# FK Audit — pre cascade fix (T0.13)")
    lines.append("")
    lines.append(
        "Read-only audit of every FK in the `public` schema. Source: "
        "`information_schema.referential_constraints` JOIN "
        "`key_column_usage` + `constraint_column_usage`. Generated by "
        "`backend-new/scripts/audit_fks.py`."
    )
    lines.append("")
    lines.append(f"- **Total FK constraints:** {len(fks)}")
    lines.append(f"- **DELETE rule distribution:** {rule_summary}")
    lines.append(
        f"- **CASCADE → soft-deletable parent (must fix in T0.14):** "
        f"{len(cascade_to_soft)}"
    )
    lines.append(
        f"- **Soft-deletable inventory size:** {len(SOFT_DELETABLE_TABLES)} tables"
    )
    lines.append("")
    lines.append(
        "## CASCADE rules to soft-deletable parents (PRIMARY — T0.14 target)"
    )
    lines.append("")
    if cascade_to_soft:
        lines.append(
            "These FKs declare `ON DELETE CASCADE` on a parent table that is in "
            "the soft-delete inventory. A parent is never hard-deleted on the "
            "happy path (soft-delete sets `deleted_at`), but a real DELETE "
            "(admin force-purge, GDPR) would silently wipe these children. "
            "T0.14 will rewrite the listed FKs to `ON DELETE NO ACTION`; T0.15 "
            "introduces an app-level recursive soft-delete service."
        )
        lines.append("")
        lines.append(
            "| from_table.column | to_table | DELETE_RULE | UPDATE_RULE | parent_soft_deletable | constraint_name |"
        )
        lines.append("|---|---|---|---|---|---|")
        for fk in cascade_to_soft:
            lines.append(_md_row(fk))
    else:
        lines.append(
            "_None — no CASCADEs target a soft-deletable parent. T0.14 has nothing to rewrite._"
        )
    lines.append("")
    lines.append("## Other FK rules (informational)")
    lines.append("")
    lines.append(
        "Includes CASCADEs to hard-delete parents (e.g. `users`, jobs, audit "
        "rows), `SET NULL`, `RESTRICT`, and `NO ACTION`. Listed for "
        "completeness; T0.14 will not touch these."
    )
    lines.append("")
    lines.append(
        "| from_table.column | to_table | DELETE_RULE | UPDATE_RULE | parent_soft_deletable | constraint_name |"
    )
    lines.append("|---|---|---|---|---|---|")
    for fk in other:
        lines.append(_md_row(fk))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- {len(cascade_to_soft)} CASCADE FK(s) target a soft-deletable parent "
        f"and require rewrite to `NO ACTION` in T0.14."
    )
    lines.append(
        f"- {len(fks) - len(cascade_to_soft)} FK(s) are out of scope for the "
        f"cascade-fix migration."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        fks = fetch_fks()
    except psycopg.OperationalError as exc:
        sys.stderr.write(
            f"[audit_fks] could not connect to database: {exc}\n"
            "Ensure docker compose is up (port 5433) and DATABASE_URL is set.\n"
        )
        return 1

    report = render_report(fks)

    repo_root = Path(__file__).resolve().parent.parent.parent
    out_path = repo_root / "docs" / "fk-audit-pre-cascade-fix.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    cascade_to_soft = sum(
        1
        for fk in fks
        if fk.delete_rule == "CASCADE" and fk.parent_soft_deletable
    )
    sys.stdout.write(
        f"[audit_fks] wrote {out_path} "
        f"({len(fks)} FKs total, {cascade_to_soft} CASCADE→soft-deletable)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
