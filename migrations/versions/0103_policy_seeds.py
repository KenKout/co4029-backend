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
        "## I. Purpose and scope\n\nThis policy explains how Learning Programs are created, published, assigned, completed, changed, and reported on the platform. It applies uniformly to every organisation, faculty, department, administrator, Faculty Dean, manager, and student using Learning Programs.\n\nA Learning Program is an academic container made up of one or more Career Paths. A Career Path may be included in more than one Learning Program.\n\n## II. Roles and responsibilities\n\n- **Platform and organisation administrators** configure system, organisation, account, and participation limits. They do not directly manage a student's academic enrolment or approve academic path changes.\n- **Faculty Deans** manage Learning Programs within their faculty scope, enrol students, and review Career Path change requests.\n- **Managers** may create, edit, and publish Learning Programs and enrol students when their assigned scope permits it. They cannot choose a student's first Career Path.\n- **Students** review the available paths in their pinned Program version and choose their own initial Career Path.\n\nAll actions are subject to active account, role, organisation, and faculty assignments.\n\n## III. Program creation and publication\n\nA Learning Program must have a name, a unique slug within its applicable scope, a faculty, and at least one published Career Path before it can be published.\n\nA draft may be edited freely. Publishing creates a frozen version. A published version cannot be changed or have a Career Path removed. Further changes require a new draft version. Existing students remain attached to the Program version used when they enrolled; a later version applies only to later enrolments unless an explicit migration is introduced.\n\nPublishing, archiving, withdrawing a student, and other one-way operations require explicit confirmation in the user interface.\n\n## IV. Student enrolment\n\nStudents are enrolled into a Learning Program, not directly into a Career Path. A Program must be published and not archived before accepting new enrolments.\n\nThe maximum number of concurrent Learning Program enrolments is configured at organisation level. The platform default is one. Enrolments awaiting path selection and active enrolments count toward this limit; completed enrolments do not.\n\nA student may participate in more than one Program at the same time only when the organisation's configured limit allows it. Course progress and completion awards are shared across Programs, while each Program Enrollment keeps its own status and completion history.\n\n## V. Initial Career Path selection\n\nAfter enrolment, the student must select one Career Path from the exact Program version assigned to them. There can be only one active Career Path for each Program Enrollment.\n\nThe first selection belongs to the student. Managers, Faculty Deans, and administrators cannot select it on the student's behalf. An archived Career Path cannot be selected by a new student.\n\n## VI. Career Path change requests\n\nAn active student may request a change from the current Career Path to another Career Path contained in the same pinned Program version, subject to all of the following:\n\n- the target differs from the current Career Path;\n- the target is not archived;\n- the Program Enrollment and current path are not completed;\n- there is no other pending request for that Program Enrollment;\n- the number of previously approved changes is below the Program version's limit, which defaults to three.\n\nThe request must include a reason. Submitting a request does not change the active path.\n\nA submitted request stays open until it is decided or withdrawn. The student may withdraw their own open request at any point before a decision, including while it is under review.\n\n## VII. Review and decision\n\nOnly an active Faculty Dean within the Program's faculty scope may act on a pending request. A reviewer cannot approve their own request.\n\nA request passes through the following states:\n\n- **Awaiting review** \u2014 submitted and not yet picked up.\n- **Under review** \u2014 a Faculty Dean has acknowledged the request and is verifying the student's record. This is a signal, not a decision: the active Career Path, progress, and change budget are unchanged, and the student is notified that their request has been received. A request may be decided with or without this step.\n- **Approved**, **rejected**, **withdrawn**, or **invalidated** \u2014 final.\n\nA request that is awaiting review or under review is an open request. Only one open request may exist for a Program Enrollment at a time.\n\nBefore approval, the platform rechecks the student's status, current path, change limit, and target path. If the target is archived while the request is pending, the request is invalidated with that reason.\n\nA rejection must state a reason from a defined set, which the student receives:\n\n- the stated justification is not sufficient;\n- too much progress would be lost by switching at this time;\n- the target Career Path is not a suitable fit for the student's record;\n- the student's remaining change should be kept for a more necessary switch;\n- an advising conversation is required before a switch;\n- supporting information for the request is missing;\n- another reason, which must be explained in writing.\n\nRejection leaves the current path unchanged and does not consume one of the student's approved changes. The reason, any written note, the reviewer, and the decision time are retained on the request record and remain visible to the student.\n\nOn approval, the old path attempt is closed with a progress snapshot and a new path attempt begins. The Program report retains the current path and the full transition history.\n\nThe student is notified when their request is acknowledged as under review, approved, or rejected.\n\n## VIII. Progress preservation and course access\n\nA course completed by a student remains completed across Learning Programs and Career Paths. If the completed course also appears in the new path, it is counted immediately toward that path.\n\nWhen a path changes, access to shared in-progress courses continues through the new path. Access to an old-only incomplete course may end when no other active Program or Career Path grants access to it. A completed course record is not removed by a path change or withdrawal.\n\n## IX. Completion\n\nCompleting the active Career Path completes that Program Enrollment. Completion is evaluated against the exact Career Path version pinned to the active attempt.\n\nCompleting one Program Enrollment does not automatically complete another Program Enrollment, even when the two Programs contain the same Career Path. Each Program retains its own version, status, and history.\n\nA completed Program Enrollment cannot change Career Path and cannot be withdrawn through the ordinary withdrawal process. Any exceptional correction must use a separately authorised and audited administrative procedure.\n\n## X. Archiving and continuity\n\nArchiving a Learning Program blocks new enrolments but does not interrupt students already enrolled. Archiving a Career Path blocks new selections and new switch approvals to that path, while students already active on it may continue under their pinned version.\n\n## XI. Records, reporting, and transparency\n\nProgram reports may include enrolment status, pinned Program version, current Career Path, current progress, approved change count, open requests and their review state, rejected requests with their recorded reason, and transition history.\n\nHistorical path attempts retain the snapshot recorded when the student left that path. Later course completions do not rewrite the historical snapshot, although they may count toward the current path through shared completion awards.\n\nAccess to individual student records is restricted by role, organisation, and faculty scope and remains subject to the [Privacy Policy](/policy/privacy).\n\n## XII. Requests, corrections, and complaints\n\nStudents should first contact the Faculty Dean or authorised academic manager responsible for their Learning Program regarding enrolment, path selection, path change, completion, or reporting concerns. Account, privacy, or platform operation concerns should be directed to the organisation administrator or the platform's official support channel.\n\nThe requester may be asked to provide the Program, relevant dates, and supporting information so the issue can be verified. Decisions and corrections that alter academic records must be authorised and auditable.\n\n## XIII. Changes to this policy\n\nThis policy may be updated when platform functionality, academic procedures, or applicable requirements change. The current version and its last-updated date will be published on this page. Material changes should be communicated through the platform before they take effect.\n",
    ),
    (
        "career-path",
        "academic",
        "Career Path Policy",
        "Converted from the platform help pages; pending admin review.",
        False,
        "## I. Purpose and scope\n\nThis policy explains how Career Paths are authored, published, included in Learning Programs, selected, followed, completed, changed, and reported on the platform. It applies uniformly across all organisations, faculties, and departments.\n\nA Career Path is a versioned sequence of stages and courses designed to guide a student toward a defined learning or career outcome.\n\n## II. Authoring responsibility\n\nOnly authorised managers and Faculty Deans may create or edit Career Paths within their assigned organisation and faculty scope. Administrators configure the platform and accounts but do not directly assign students to Career Paths or make academic decisions for them.\n\nA Career Path must have a name and slug. Its stages, courses, required-course flags, ordering, and unlock rules must be reviewable before publication.\n\n## III. Drafts, publication, and versions\n\nA draft Career Path may be edited before publication. Publishing freezes the version, including its course membership, stage ordering, required-course rules, and progression settings.\n\nA published version cannot be deleted or changed in place. Further changes require a new draft version. A new version does not retroactively change a Learning Program version, Program Enrollment, or Path Attempt that already points to an earlier version.\n\nWhere a draft contains unpublished course dependencies, the author must explicitly resolve them before completing publication.\n\n## IV. Use in Learning Programs\n\nA Career Path may appear in multiple Learning Programs. Each Program version stores the exact published Career Path version selected by the author; it does not automatically follow later Career Path versions.\n\nStudents are not directly enrolled into a Career Path by a manager. They receive access through a Learning Program Enrollment and select one of the paths included in their pinned Program version.\n\n## V. Path selection and active status\n\nA student may have one active Career Path for each Program Enrollment. Participation in another Program may create another active path without replacing the first, subject to the organisation's concurrent Program limit.\n\nThe student chooses the initial path. A later change requires the formal request and Faculty Dean review described in the [Learning Program Policy](/policy/learning-program).\n\n## VI. Stages and progression\n\nThe first stage is always available. Later stages follow the unlock rule stored in the published Career Path version, such as always available, available after progress in the previous stage, or available after the previous stage's required work is satisfied.\n\nUnknown or invalid progression rules fail closed: the platform must not silently unlock restricted learning content.\n\nReordering stages in a draft does not silently rewrite their configured unlock rules. Authors are responsible for reviewing warnings caused by a changed stage position before publication.\n\n## VII. Required courses and completion\n\nCareer Path completion is based on the required courses in the student's pinned path version. A non-empty path is completed when all required courses are satisfied under the published rules.\n\nOptional courses may support learning and remain visible where access permits, but they do not block completion unless the published version explicitly marks them as required.\n\nCompleting the active Career Path completes the corresponding Learning Program Enrollment.\n\n## VIII. Shared course completion\n\nCourse completion belongs to the student and course, not only to one Career Path. Once awarded, completion may satisfy the same course in another Career Path or Learning Program.\n\nChanging paths does not erase completed courses. A later reduction in mutable progress does not automatically remove an awarded completion record. Any exceptional reversal must be separately authorised and audited.\n\nPath progress is always calculated against the exact published Career Path version pinned to the student's Path Attempt.\n\n## IX. Path changes\n\nStudents cannot directly replace their active path. They may request a change only within the same pinned Learning Program version and only while the current Program Enrollment remains active and incomplete.\n\nWhen a change is approved:\n\n- the old Path Attempt stops at a frozen exit snapshot;\n- the new Path Attempt starts a separate timeline;\n- shared completed courses count toward the new path;\n- shared in-progress course access is preserved;\n- old-only incomplete course access may end when no other active entitlement grants it.\n\nThe old and new attempts remain available in authorised academic reporting.\n\n## X. Archiving\n\nA published Career Path is archived rather than deleted. Archiving prevents new Program drafts from adding it, prevents new students from selecting it, and invalidates pending changes targeting it.\n\nStudents already active on the archived path may continue under their pinned version. Archiving does not rewrite their progress, completion awards, or history.\n\n## XI. Reporting and readiness\n\nCurrent progress and readiness are calculated from the current active Path Attempt and its pinned version. A switched-out, cancelled, or completed attempt is reported using its exit snapshot rather than recalculating its old percentage from later activity.\n\nIndividual student records are visible only to authorised roles within the relevant organisation and faculty scope and remain subject to the [Privacy Policy](/policy/privacy).\n\n## XII. Requests, corrections, and complaints\n\nStudents should contact the responsible Faculty Dean or authorised academic manager if they believe a path, course requirement, progress result, archive status, or transition history is incorrect. The requester may be asked to identify the Learning Program, Career Path, affected course, and relevant dates.\n\nChanges to published academic history require an authorised, recorded correction process. Support personnel must not silently edit frozen versions or historical snapshots.\n\n## XIII. Changes to this policy\n\nThis policy may be revised when Career Path functionality, academic procedures, or applicable requirements change. The current text and last-updated date will be published on this page. Material changes should be communicated through the platform before taking effect.\n",
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
