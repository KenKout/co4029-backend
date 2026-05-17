"""Positive-case fixture for Pattern 2 -- bulk Core DML on protected model.

This file is INTENTIONALLY non-compliant. ``Course`` carries both
``AuditedByMixin`` and ``SoftDeleteMixin``; ``session.execute(sa.delete(Course))``
LOOKS like ORM but bypasses the unit-of-work and the audit listener
never fires. The lint test (``test_bulk_dml_caught``) asserts the
scanner detects this exact shape.

A local stub class named ``Course`` is used so this file does not import
``abridgeai.features.courses.models`` directly (which would itself cross
a feature boundary). The scanner resolves ``$M`` against the live
``Base.registry`` -- which DOES contain the real ``Course``, mapped to
table ``courses`` -- so the violation fires regardless of the local
stub's identity.
"""

from __future__ import annotations

import sqlalchemy as sa


class Course:
    pass


def _bad(session: object) -> None:
    session.execute(sa.delete(Course))  # type: ignore[attr-defined]
