"""Seed the platform policy catalogue.

Five documents, matching the slugs the front end has advertised since the
static help pages (``help-content.ts`` POLICY_ORDER):

* ``terms`` / ``privacy`` / ``cookies`` — the fixed legal set. Seeded
  PUBLISHED so the footer and landing links resolve from the first boot; the
  bodies are skeletons for legal review, versioned like any other policy.
* ``learning-program`` / ``career-path`` — the academic-operational set,
  converted from the front-end constants to markdown. Seeded as DRAFTS: an
  admin reviews and publishes them, which is exactly what the authoring UI
  is for.

Audience: the three legal documents name the STUDENT role — the universal
audience, every reader is a party to them (see queries.STUDENT_ROLE_CODE).
The academic drafts ship with no audience rows (public on publish); an admin
narrows them before publishing if needed.

Idempotent: every insert is guarded by slug existence, so re-running the
migration (or running it on a DB where an admin already recreated a policy
with the same slug) skips rather than duplicating. The unique partial index
``uq_policies_slug`` is the hard backstop.

Revision ID: 0103_policy_seeds
Revises: 0102_policies
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision = "0103_policy_seeds"
down_revision = "0102_policies"
branch_labels = None
depends_on = None


# (slug, category, title, changelog, published, body)
SEEDS: list[tuple[str, str, str, str, bool, str]] = [
    (
        "terms",
        "legal",
        "Terms of Service",
        "Initial release.",
        True,
        """## 1. Acceptance of terms

By creating an account or otherwise using aBridgeAI you agree to these
Terms of Service. If you do not agree, do not use the platform.

## 2. The service

aBridgeAI is a learning platform that hosts courses, AI-assisted content
generation, assessments and interviews. Features may evolve; material
changes to the service will be announced.

## 3. Accounts

You are responsible for the activity under your account and for keeping
your credentials secure. One account per person; do not share access.

## 4. Content and conduct

- Upload only content you have the right to use.
- Do not attempt to disrupt, probe or overload the platform.
- Follow your organization's academic-integrity rules when taking
  assessments.

## 5. Availability

We aim for high availability but do not guarantee uninterrupted service.
Scheduled maintenance is announced in advance where practical.

## 6. Termination

We may suspend or terminate accounts that violate these terms. You may stop
using the platform at any time.

## 7. Changes to these terms

Material changes are published as a new version of this document with a new
effective date. Continued use after the effective date constitutes
acceptance.

## 8. Contact

Questions about these terms: contact your platform administrator.
""",
    ),
    (
        "privacy",
        "legal",
        "Privacy Policy",
        "Initial release.",
        True,
        """## 1. What we collect

- **Account data** — name, email, organizational affiliation.
- **Learning data** — enrolments, lesson progress, quiz and interview
  attempts, review history.
- **Technical data** — request logs, session records, necessary cookies.

## 2. How we use it

- To operate the platform: enrolment, progress tracking, assessments.
- To keep it secure: audit logs, abuse detection.
- To communicate platform events you are subscribed to.

## 3. What we do not do

- We do not sell personal data.
- We do not use your learning content to train models outside your
  organization's own workspace.

## 4. AI processing

Materials you upload for AI processing are processed to produce learning
content for your organization. Processing telemetry (model, token counts,
cost) is retained for administration; prompts and outputs follow the same
retention rules as the content they belong to.

## 5. Your rights

You may request a copy or deletion of your personal data through your
administrator. Enrollment and progress records tied to academic records are
retained per your organization's policy.

## 6. Contact

Privacy questions: contact your platform administrator.
""",
    ),
    (
        "cookies",
        "legal",
        "Cookie Policy",
        "Initial release.",
        True,
        """## What we set

aBridgeAI uses a minimal set of cookies and equivalent storage:

- **Session cookie** — keeps you signed in. Essential; the platform cannot
  function without it.
- **Preferences** — language and interface preferences stored locally.

## What we do not set

No advertising cookies, no third-party trackers, no cross-site profiling.

## Managing cookies

You can clear cookies from your browser at any time; you will simply be
signed out. Because the only essential cookie is the session cookie, there
is no consent banner to manage.
""",
    ),
    (
        "learning-program",
        "academic",
        "Learning Program Policy",
        "Converted from the platform help pages; pending admin review.",
        False,
        """## Purpose

This policy governs participation in structured **Learning Programs** —
curated bundles of courses that an organization assigns as a track of
study.

## Enrolment

- You are enrolled in a Learning Program by your organization or you join
  one that is open.
- Enrolment in a program does not automatically enrol you in every course
  inside it; courses are joined individually as you progress.

## Progress and completion

- A program tracks your progress across its stages. Completing a course
  advances the stage it belongs to.
- Programs may define required and elective stages; the program page shows
  which is which.

## Switching

- Requesting a switch to another Learning Program is subject to review by
  your organization. Your course enrolments and progress are not deleted by
  a switch.

## Responsibilities

- Learners: follow the program sequence; assessments inside courses remain
  subject to the course's own integrity rules.
- Managers: review switch requests and maintain program curriculum.
""",
    ),
    (
        "career-path",
        "academic",
        "Career Path Policy",
        "Converted from the platform help pages; pending admin review.",
        False,
        """## Purpose

This policy governs **Career Paths** — the role-oriented view of learning
that maps courses onto the stages of a target role.

## Stages

- A Career Path is an ordered sequence of stages. Each stage groups courses
  relevant to that point of the journey.
- Your position on a path ("Stage N") is derived from the courses you have
  joined, not set manually.

## Enrolment

- Joining a Career Path is open unless your organization restricts it.
- A course may appear on several paths; joining it once counts for every
  path that references it.

## Changes

- Managers may reorder stages or move courses between them. Such changes
  take effect for the path as a whole; your personal progress on individual
  courses is never affected.
- Changes to a path's structure are visible immediately to its members.

## Responsibilities

- Learners: use the path as guidance; your own learning goals take
  precedence over the suggested order.
- Managers: keep paths current, and review requests for structural changes.
""",
    ),
]


def upgrade() -> None:
    conn = op.get_bind()
    now = datetime.now(tz=UTC)

    student_role_id = conn.execute(
        sa.text("SELECT id FROM roles WHERE code = 'student'")
    ).scalar()

    for slug, category, title, changelog, published, body in SEEDS:
        exists = conn.execute(
            sa.text("SELECT 1 FROM policies WHERE slug = :slug AND deleted_at IS NULL"),
            {"slug": slug},
        ).scalar()
        if exists:
            continue

        policy_id = uuid.uuid4()
        conn.execute(
            sa.text(
                "INSERT INTO policies (id, slug, category, created_at, updated_at) "
                "VALUES (:id, :slug, :category, :now, :now)"
            ),
            {"id": policy_id, "slug": slug, "category": category, "now": now},
        )
        conn.execute(
            sa.text(
                "INSERT INTO policy_versions (id, policy_id, version_no, language, "
                "status, title, body, format, changelog, published_at, created_at, updated_at) "
                "VALUES (:id, :policy_id, 1, 'en', :status, :title, :body, 'markdown', "
                ":changelog, :published_at, :now, :now)"
            ),
            {
                "id": uuid.uuid4(),
                "policy_id": policy_id,
                "status": "published" if published else "draft",
                "title": title,
                "body": body,
                "changelog": changelog,
                "published_at": now if published else None,
                "now": now,
            },
        )
        if student_role_id is not None:
            conn.execute(
                sa.text(
                    "INSERT INTO policy_audience_roles (id, policy_id, role_id, created_at, updated_at) "
                    "VALUES (:id, :policy_id, :role_id, :now, :now)"
                ),
                {"id": uuid.uuid4(), "policy_id": policy_id, "role_id": student_role_id, "now": now},
            )


def downgrade() -> None:
    slugs = [s[0] for s in SEEDS]
    conn = op.get_bind()
    # Only rows this migration created are removed: an admin-published
    # revision of `terms` keeps its slug but has different content, so match
    # on the seeded changelog rather than the slug alone.
    conn.execute(
        sa.text(
            "DELETE FROM policy_audience_roles WHERE policy_id IN ("
            " SELECT p.id FROM policies p JOIN policy_versions v ON v.policy_id = p.id"
            " WHERE p.slug = ANY(:slugs) AND v.changelog IN ('Initial release.',"
            " 'Converted from the platform help pages; pending admin review.'))"
        ),
        {"slugs": slugs},
    )
    conn.execute(
        sa.text(
            "DELETE FROM policy_versions WHERE changelog IN ('Initial release.',"
            " 'Converted from the platform help pages; pending admin review.')"
        ),
    )
    conn.execute(
        sa.text("DELETE FROM policies WHERE slug = ANY(:slugs)"),
        {"slugs": slugs},
    )
