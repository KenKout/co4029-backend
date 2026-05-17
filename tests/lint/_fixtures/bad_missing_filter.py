"""Positive-case fixture for Pattern 3 -- raw SELECT missing deleted_at filter.

This file is INTENTIONALLY non-compliant. ``courses`` is a SoftDeleteMixin
table; the SELECT below references it without ``deleted_at IS NULL``,
so soft-deleted rows would leak. The lint test
(``test_missing_filter_caught``) asserts the scanner detects this exact
shape.
"""

from __future__ import annotations

from sqlalchemy import text


def _bad() -> None:
    text("SELECT id, title FROM courses WHERE organization_id = :id")
