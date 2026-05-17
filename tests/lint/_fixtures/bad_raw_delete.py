"""Positive-case fixture for Pattern 1 -- raw DELETE on a soft-delete table.

This file is INTENTIONALLY non-compliant. ``courses`` is a SoftDeleteMixin
table; raw ``text("DELETE FROM courses ...")`` would bypass the
hard-delete guard listener. The lint test
(``test_raw_delete_caught``) asserts the scanner detects this exact
shape; if you remove the line, the positive-case test fails.
"""

from __future__ import annotations

from sqlalchemy import text


def _bad() -> None:
    text("DELETE FROM courses WHERE id = :id")
