"""Add course contact-info columns (teacher-published, learner-visible)

Revision ID: 0061_course_contact
Revises: 0060_persona_profile
Create Date: 2026-07-25

Lets a teacher publish contact details on the course landing page students see:

* ``contact_email``   VARCHAR(320) — the teacher's contact address. The SPA
  pre-fills this from the teacher's account email, but it's a plain editable
  column (a teacher may want a different public address than their login).
* ``contact_phone``   VARCHAR(50)  — free-form phone; no format constraint so
  international / extension formats aren't rejected.
* ``contact_website_url`` VARCHAR(500) — personal / course website.
* ``contact_social_url``  VARCHAR(500) — a social-media profile link.

All four are nullable and default NULL, so every existing course is unaffected
and simply shows no contact block until the teacher fills one in. URLs are
length-bounded but not CHECK-validated at the DB — validation lives in the
Pydantic request schema (``AnyUrl``/normalisation) so the rules stay editable
without a migration.

Additive, nullable columns — fully reversible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# NOTE: alembic_version.version_num is varchar(32); keep revision id <= 32 chars.
revision = "0061_course_contact"
down_revision = "0060_persona_profile"
branch_labels = None
depends_on = None

_TABLE = "courses"
_COLUMNS = (
    ("contact_email", sa.String(length=320)),
    ("contact_phone", sa.String(length=50)),
    ("contact_website_url", sa.String(length=500)),
    ("contact_social_url", sa.String(length=500)),
)


def upgrade() -> None:
    for name, type_ in _COLUMNS:
        op.add_column(_TABLE, sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name, _type in reversed(_COLUMNS):
        op.drop_column(_TABLE, name)
