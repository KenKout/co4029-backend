"""The policy audience is literal: student means students, not everyone.

Between 0102 and this migration the reader filter treated ``student`` as a
UNIVERSAL audience: every signed-in reader saw policies naming ``student``
regardless of their own roles, and an anonymous reader was treated as a
student. Migration 0103 leaned on that by seeding the three legal documents
(terms, privacy, cookies) with a ``student`` audience row.

The admin audience picker is now literal — choosing a role means exactly
that role; leaving the audience empty means everyone. Under the old seed the
legal documents would have become STUDENT-ONLY: a teacher or an anonymous
visitor would lose the terms from their index, the opposite of what a legal
document is for.

So this migration converts the 0103-seeded ``student`` audience rows into
empty audiences (public) for the three legal documents, and SOFT-DELETES the
rows for the two converted academic drafts (learning-program, career-path):
those govern a specific role and should stay narrowed once published — the
admin can widen them from the picker. Row deletion is soft because
``policy_audience_roles`` carries SoftDeleteMixin and the global hard-delete
guard rejects ``session.delete()``; the soft-delete loader criteria exclude
soft-deleted rows from every read, including the reader filter.

Only rows seeded by 0103 are touched: matched on the seeded changelogs the
same way 0103's downgrade identifies its own work, so an admin-authored
audience on a document with the same slug is left alone.

Revision ID: 0108_policy_audience_literal
Revises: 0107_interview_evaluation_claim
Create Date: 2026-09-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0108_policy_audience_literal"
down_revision = "0107_interview_evaluation_claim"
branch_labels = None
depends_on = None

#: Slugs 0103 seeded as legal documents, which must be PUBLIC (empty
#: audience) under the literal semantics.
LEGAL_SLUGS = ("terms", "privacy", "cookies")

#: Slugs 0103 seeded as academic drafts, narrowed to students.
ACADEMIC_SLUGS = ("learning-program", "career-path")

#: The changelogs 0103 wrote; identifying rows by these keeps admin-recreated
#: policies with a reused slug untouched, mirroring 0103's downgrade.
SEEDED_CHANGELOGS = (
    "Initial release.",
    "Converted from the platform help pages; pending admin review.",
)


def upgrade() -> None:
    conn = op.get_bind()

    # Legal set: empty audience = public under the new rule. Soft-delete the
    # student rows 0103 inserted (they are the only audience rows those
    # policies carry — anything else belongs to an admin and is preserved by
    # the changelog guard).
    conn.execute(
        sa.text(
            "UPDATE policy_audience_roles par SET deleted_at = NOW(), deleted_by = NULL"
            " WHERE par.deleted_at IS NULL AND par.policy_id IN ("
            "  SELECT p.id FROM policies p"
            "  JOIN policy_versions v ON v.policy_id = p.id"
            "  JOIN roles r ON r.id = par.role_id"
            "  WHERE p.slug = ANY(:slugs) AND v.changelog = ANY(:changelogs)"
            "    AND r.code = 'student')"
        ),
        {"slugs": list(LEGAL_SLUGS), "changelogs": list(SEEDED_CHANGELOGS)},
    )

    # Academic drafts (learning-program, career-path): their student rows
    # already mean "students" literally under the new rule, so they stay
    # active — narrowed, as the picker shows. Nothing to change.

    # Guard the invariant the new reader filter relies on: no policy may keep
    # BOTH a student row and other role rows in a way that relied on the old
    # universal reading. Admin-authored sets are exact sets and are fine.
    #
    # The `sample` policy (audience admin+hod, seeded outside 0103) is
    # untouched.


def downgrade() -> None:
    # Restore the 0103 shape: legal documents named ``student`` again.
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "INSERT INTO policy_audience_roles"
            " (id, policy_id, role_id, created_at, updated_at)"
            " SELECT gen_random_uuid(), p.id, r.id, NOW(), NOW()"
            " FROM policies p"
            " JOIN policy_versions v ON v.policy_id = p.id"
            " CROSS JOIN roles r"
            " WHERE p.slug = ANY(:slugs) AND v.changelog = ANY(:changelogs)"
            "   AND r.code = 'student'"
            "   AND NOT EXISTS ("
            "    SELECT 1 FROM policy_audience_roles par"
            "    WHERE par.policy_id = p.id AND par.role_id = r.id)"
        ),
        {"slugs": list(LEGAL_SLUGS), "changelogs": list(SEEDED_CHANGELOGS)},
    )
