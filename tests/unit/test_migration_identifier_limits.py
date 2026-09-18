"""Migration identifiers that Postgres will actually accept.

Two separate limits, and both fail late — at ``alembic upgrade`` against a
real database, long after review has passed.

**Revision ids: 32 characters.** Alembic stores the applied revision in
``alembic_version.version_num``, which it creates as ``VARCHAR(32)`` unless a
project overrides it. ``migrations/env.py`` does not. A longer id writes fine
in Python, is accepted by every local check, and then fails on the INSERT
that records the upgrade — after the schema change has already been applied,
which is the worst moment to discover it.

**Everything else: 63 characters.** Table, column, index and constraint names
are Postgres identifiers, silently TRUNCATED to ``NAMEDATALEN - 1``. That is
worse than an error: two names sharing a 63-character prefix collapse into
one, and a ``DROP`` in the downgrade then names something that does not
exist.

Both have been hit here. ``0129`` shipped at 37 characters and ``0132`` at
35, each needing a follow-up commit to shorten. This file is cheaper than the
next one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: Width of ``alembic_version.version_num`` as Alembic creates it.
_REVISION_MAX = 32

#: Postgres ``NAMEDATALEN - 1``. Longer identifiers are truncated, not refused.
_IDENTIFIER_MAX = 63

_VERSIONS = Path(__file__).resolve().parents[2] / "migrations" / "versions"

# Both declaration styles in the tree: a bare assignment, and the annotated
# form ``revision: str = "..."`` / ``down_revision: str | None = "..."`` that
# 0127 uses. A pattern that matched only the first would skip that file
# silently -- and a guard with a hole in it is worse than no guard, because
# the hole is invisible until something slips through it.
_REVISION_RE = re.compile(
    r"""^revision \s* (?::[^=]+)? = \s* ["']([^"']+)["']""", re.M | re.X
)
_DOWN_RE = re.compile(
    r"""^down_revision \s* (?::[^=]+)? = \s* ["']([^"']+)["']""", re.M | re.X
)

# Names passed to op.create_index / create_unique_constraint /
# create_foreign_key / create_check_constraint, and the ones declared inline.
_NAME_RE = re.compile(r'name="([a-z][a-z0-9_]{3,})"|"((?:uq|ix|ck|fk)_[a-z0-9_]+)"')


def _migrations() -> list[Path]:
    files = sorted(p for p in _VERSIONS.glob("*.py") if p.name != "__init__.py")
    assert files, f"no migrations found under {_VERSIONS}"
    return files


def test_every_migration_declares_a_revision() -> None:
    """Guards the guard.

    Every check below reads the revision with a regex. A file whose
    declaration style the pattern does not match would be skipped in silence,
    and the suite would report a clean scan of a tree it never fully read.
    """
    missing = [
        path.name
        for path in _migrations()
        if _REVISION_RE.search(path.read_text(encoding="utf-8")) is None
    ]

    assert missing == [], f"no revision id could be read from: {missing}"


@pytest.mark.parametrize("path", _migrations(), ids=lambda p: p.name)
def test_revision_ids_fit_the_alembic_version_column(path: Path) -> None:
    """A revision longer than the column fails AFTER the schema changed."""
    source = path.read_text(encoding="utf-8")

    for match in (_REVISION_RE.search(source), _DOWN_RE.search(source)):
        if match is None:
            continue
        identifier = match.group(1)
        assert len(identifier) <= _REVISION_MAX, (
            f"{path.name}: revision id {identifier!r} is {len(identifier)} "
            f"characters; alembic_version.version_num holds {_REVISION_MAX}"
        )


@pytest.mark.parametrize("path", _migrations(), ids=lambda p: p.name)
def test_constraint_and_index_names_fit_a_postgres_identifier(path: Path) -> None:
    """Over 63 characters Postgres truncates rather than refusing.

    The create succeeds under a name nobody wrote down, and the matching drop
    in ``downgrade`` then refers to an object that does not exist.
    """
    source = path.read_text(encoding="utf-8")

    for match in _NAME_RE.finditer(source):
        identifier = match.group(1) or match.group(2)
        assert len(identifier) <= _IDENTIFIER_MAX, (
            f"{path.name}: identifier {identifier!r} is {len(identifier)} "
            f"characters; Postgres truncates at {_IDENTIFIER_MAX}"
        )


def test_every_revision_is_unique() -> None:
    """Two migrations claiming one id gives Alembic an ambiguous head, and
    the second to be applied is recorded as the first."""
    seen: dict[str, str] = {}
    for path in _migrations():
        match = _REVISION_RE.search(path.read_text(encoding="utf-8"))
        if match is None:
            continue
        identifier = match.group(1)
        assert identifier not in seen, (
            f"{path.name} and {seen[identifier]} both declare revision {identifier!r}"
        )
        seen[identifier] = path.name


def test_the_chain_has_exactly_one_head() -> None:
    """A second head means two migrations share a parent, and `upgrade head`
    becomes ambiguous — the failure reads as a merge conflict long after the
    branch that caused it."""
    revisions: set[str] = set()
    parents: set[str] = set()
    for path in _migrations():
        source = path.read_text(encoding="utf-8")
        revision = _REVISION_RE.search(source)
        if revision is None:
            continue
        revisions.add(revision.group(1))
        down = _DOWN_RE.search(source)
        if down is not None:
            parents.add(down.group(1))

    heads = revisions - parents
    assert len(heads) == 1, f"expected one head, found {sorted(heads)}"
