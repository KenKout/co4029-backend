"""The LIKE-escaping behind the HTTP-audit path filter.

The filter used to be a PREFIX match against the full stored path
(``/api/v1/admin/audit/http``), so an operator typing ``admin`` — or even
``/admin`` — got nothing back and the search box looked broken. It is now a
case-insensitive substring, which means the operator's text reaches a ``LIKE``
pattern and its metacharacters have to be neutralised first.

Pure function, no database: this is the part that decides whether a typed
underscore is an underscore.
"""

from __future__ import annotations

import pytest

from abridgeai.features.admin.queries.audit import _escape_like

BACKSLASH = "\\"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Ordinary terms pass through untouched — the common case.
        ("admin/audit", "admin/audit"),
        ("api/v1/me", "api/v1/me"),
        # Trimmed, because a trailing space from a paste is not a search term.
        ("  admin  ", "admin"),
        # Nothing to filter on collapses to NULL, which the SQL reads as
        # "no path condition" rather than "match the empty string".
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_ordinary_terms(raw: str | None, expected: str | None) -> None:
    assert _escape_like(raw) == expected


def test_underscore_is_a_literal_not_a_wildcard() -> None:
    """``_`` matches any single character in LIKE, and paths are full of them."""
    assert _escape_like("quiz_attempts") == "quiz" + BACKSLASH + "_attempts"


def test_percent_is_a_literal_not_a_wildcard() -> None:
    """An unescaped ``%`` would match the entire table."""
    assert _escape_like("100%") == "100" + BACKSLASH + "%"


def test_backslash_is_escaped_before_the_characters_it_would_escape() -> None:
    """Order matters, and getting it wrong is silent.

    Escaping ``%`` and ``_`` first would leave their new backslashes to be
    escaped again by the backslash pass, turning the escape into a literal and
    handing the wildcard back to the pattern.
    """
    assert _escape_like(BACKSLASH) == BACKSLASH * 2
    assert _escape_like(BACKSLASH + "_") == BACKSLASH * 3 + "_"
