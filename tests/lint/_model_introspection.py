"""Static AST scanner for raw-SQL audit-bypass detection (T4).

This module is the shared scan engine used by both the pytest lint test
(``tests/lint/test_no_audit_bypass.py``) and the CLI gate
(``tools/audit_raw_sql.py``). It detects three anti-patterns that would
silently bypass the project's audit / soft-delete / hard-delete safety
nets:

1. **Pattern ``raw_delete``** -- ``text("DELETE FROM <soft_delete_table>")``.
   Even though the hard-delete guard listener fires for ORM ``Session``
   deletes, raw ``text()`` ``DELETE`` runs at the Core layer and the
   guard never sees it. The single legitimate site is materials'
   ``_storage.purge_chunks_for_version`` (``document_chunks`` is derived
   data, intentionally not a soft-delete table).

2. **Pattern ``bulk_dml``** -- ``session.execute(sa.delete(M))`` /
   ``session.execute(sa.update(M))`` (or the bare ``delete``/``update``
   import style) where ``M`` is a model carrying ``AuditedByMixin`` or
   ``SoftDeleteMixin``. Bulk Core DML LOOKS like ORM (it takes a model
   class) but it bypasses the unit-of-work, which means the audit
   listener never fires and ``deleted_at`` is never stamped. Zero
   tolerance: every audited / soft-deletable model must be mutated via
   per-row ORM operations or the explicit ``soft_delete_cascade()``
   helper.

3. **Pattern ``missing_filter``** -- ``text("SELECT ... FROM <soft_delete_table>")``
   missing ``deleted_at IS NULL``. The ORM-level soft-delete loader
   criteria does NOT apply to raw SQL, so any ``text()`` SELECT against
   a soft-delete table must inline the filter.

The module deliberately AVOIDS importing ``features.*.models`` (or
``abridgeai.api``): importing models from ``tests/lint/`` would itself
violate the spirit of the import-linter contracts (lint code reaching
into feature internals), AND it would force the CLI tool to depend on
the full project install (fastapi etc.) just to discover tablenames.
Instead we parse every ``models.py`` file with ``ast`` and extract the
class hierarchy / ``__tablename__`` assignments directly. This stays
honest with the import-linter contracts and keeps the CLI lightweight
enough to run under bare ``python tools/audit_raw_sql.py``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Repo geometry
# ---------------------------------------------------------------------------

# tests/lint/_model_introspection.py -> backend-new/
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ABRIDGEAI_ROOT: Final[Path] = REPO_ROOT / "abridgeai"


# ---------------------------------------------------------------------------
# Allowlists (file-relative-to-REPO_ROOT, forward-slash, no leading "./")
# ---------------------------------------------------------------------------

# Pattern 1 -- raw DELETE on soft-delete table.
# Sole legitimate site: document_chunks purge (derived data, not soft-deleted).
PATTERN1_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"abridgeai/features/materials/services/authoring/_storage.py"}
)

# Pattern 2 -- bulk Core DML on AuditedByMixin/SoftDeleteMixin models.
# Zero tolerance.
PATTERN2_ALLOWLIST: Final[frozenset[str]] = frozenset()

# Pattern 3 -- raw SELECT missing deleted_at IS NULL.
#
# Tier A -- bootstrap / system-catalog / liveness queries that
# legitimately don't need soft-delete filtering OR can't apply it
# (system catalogs, audit log writes, principal lookup before the user
# is authenticated). Permanent.
#
# Tier B -- Wave 1 baseline (T4): pre-existing raw-SQL sites whose
# `deleted_at IS NULL` filter is missing as of orm-consolidation kickoff.
# Wave 5 (soft-delete enforcement) MUST drive this set to empty; the
# CLI tool `audit_raw_sql.py --kind missing_filter` is the success-
# criteria gate that ratchets the count down each wave. Do NOT extend
# Tier B further -- new sites must inline the filter.
PATTERN3_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        # Tier A -- permanent.
        "abridgeai/core/security/__init__.py",
        "abridgeai/core/observability/audit_log.py",
        "abridgeai/ai/llm/embeddings.py",
        "abridgeai/api/healthz.py",
        # Deleting an organization must not erase the record of who changed
        # its runtime config. The LEFT JOIN there exists only to label the
        # audit row with the org's name; filtering soft-deleted orgs out would
        # blank that label precisely for the tenant an investigation is most
        # likely to be about.
        "abridgeai/features/admin/queries/setting_changes.py",
        # Tier B -- Wave 1 baseline (12 sites, T4 commit). Wave 5 target: 0.
        "abridgeai/features/career_paths/queries/published.py",
        "abridgeai/features/courses/queries/published.py",
        "abridgeai/features/courses/routers/assignment.py",
        "abridgeai/features/interviews/services/authoring.py",
        "abridgeai/features/materials/ingestion/pipeline.py",
        "abridgeai/features/materials/services/authoring/_storage.py",
        "abridgeai/features/materials/workers/ingest.py",
        "abridgeai/features/quizzes/services/authoring.py",
        "abridgeai/features/spaced_repetition/routers/teacher.py",
        "abridgeai/features/spaced_repetition/services/remediation.py",
        "abridgeai/features/spaced_repetition/services/review.py",
    }
)

# Pattern 3 also exempts every file under any feature's ``queries/sql/``
# directory: those are legitimate co-located CTE / aggregation files
# whose ``text()`` body is loaded from an external ``.sql`` file via
# ``importlib.resources.files(...).read_text()``. The wrapper call is
# purely a loader and the SQL body lives in a separate file the scanner
# doesn't read. Detected by path-prefix membership.
PATTERN3_PATH_SUBSTRING_ALLOWLIST: Final[tuple[str, ...]] = ("/queries/sql/",)


# ---------------------------------------------------------------------------
# Model introspection (Base.registry, no direct features.*.models imports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTables:
    """Tablename sets for the two protected mixins."""

    soft_delete: frozenset[str]
    audited_by: frozenset[str]

    @property
    def protected_union(self) -> frozenset[str]:
        return self.soft_delete | self.audited_by


_MIXIN_SOFT_DELETE: Final[str] = "SoftDeleteMixin"
_MIXIN_AUDITED_BY: Final[str] = "AuditedByMixin"


def _base_name(node: ast.expr) -> str | None:
    """Extract the rightmost identifier from a class-base expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _tablename_in_body(class_def: ast.ClassDef) -> str | None:
    """Find ``__tablename__ = "..."`` in the class body."""
    for stmt in class_def.body:
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
        else:
            continue
        for target in targets:
            if not (isinstance(target, ast.Name) and target.id == "__tablename__"):
                continue
            value = stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    return None


def _collect_class_defs(roots: list[Path]) -> dict[str, tuple[list[str], str | None]]:
    """Return ``{class_name: (base_names, tablename_or_None)}`` across ``roots``."""
    out: dict[str, tuple[list[str], str | None]] = {}
    for root in roots:
        for path in root.rglob("models.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = [b for b in (_base_name(b) for b in node.bases) if b is not None]
                tablename = _tablename_in_body(node)
                # Last-write-wins on duplicate names is acceptable: model
                # classes are unique by import path and we don't try to
                # resolve cross-feature collisions.
                out[node.name] = (bases, tablename)
    return out


def _has_mixin(
    class_name: str,
    mixin: str,
    classes: dict[str, tuple[list[str], str | None]],
    seen: set[str] | None = None,
) -> bool:
    """Transitive: does ``class_name`` inherit ``mixin`` directly or via parents?"""
    if seen is None:
        seen = set()
    if class_name in seen:
        return False
    seen.add(class_name)
    info = classes.get(class_name)
    if info is None:
        return False
    bases, _ = info
    if mixin in bases:
        return True
    return any(_has_mixin(b, mixin, classes, seen) for b in bases)


@lru_cache(maxsize=1)
def _all_classes() -> dict[str, tuple[list[str], str | None]]:
    return _collect_class_defs([ABRIDGEAI_ROOT])


@lru_cache(maxsize=1)
def get_model_tables() -> ModelTables:
    """Resolve protected tablenames by AST-parsing every ``models.py``."""
    classes = _all_classes()
    soft: set[str] = set()
    audited: set[str] = set()
    for name, (_, tablename) in classes.items():
        if tablename is None:
            continue
        if _has_mixin(name, _MIXIN_SOFT_DELETE, classes):
            soft.add(tablename)
        if _has_mixin(name, _MIXIN_AUDITED_BY, classes):
            audited.add(tablename)
    return ModelTables(soft_delete=frozenset(soft), audited_by=frozenset(audited))


@lru_cache(maxsize=1)
def get_model_class_to_tablename() -> dict[str, str]:
    """Map ``ClassName`` -> ``__tablename__`` for every parsed model class."""
    classes = _all_classes()
    return {name: t for name, (_, t) in classes.items() if t is not None}


# ---------------------------------------------------------------------------
# Public scan API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """A single offending site."""

    file: str  # path RELATIVE to REPO_ROOT, forward-slash
    line: int
    reason: str

    def render(self) -> str:
        return f"{self.file}:{self.line}: {self.reason}"


def python_files(root: Path) -> Iterator[Path]:
    """Yield every ``.py`` file under ``root`` excluding caches/migrations."""
    for path in root.rglob("*.py"):
        parts = path.parts
        if "__pycache__" in parts:
            continue
        if "migrations" in parts and "versions" in parts:
            # Alembic-generated migration revisions (out of scope).
            continue
        yield path


def _rel(path: Path) -> str:
    """Return the path relative to REPO_ROOT in forward-slash form."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


# --- Pattern 1: text("DELETE FROM <soft_delete_table>") --------------------

_DELETE_FROM_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*DELETE\s+FROM\s+(?:ONLY\s+)?\"?(\w+)\"?",
    re.IGNORECASE,
)


def _is_text_call(node: ast.Call) -> bool:
    """``text("...")`` (Name) OR ``sa.text("...")`` / ``something.text("...")``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "text"
    if isinstance(func, ast.Attribute):
        return func.attr == "text"
    return False


def _string_literal(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def scan_raw_delete(
    file: Path,
    *,
    soft_tables: frozenset[str],
    apply_default_allowlist: bool = True,
    match_all_tables: bool = False,
) -> list[Violation]:
    """Find ``text("DELETE FROM <soft_delete_table>")`` sites.

    With ``match_all_tables=True`` every raw ``text()`` DELETE is
    reported regardless of the target table. The CLI gate uses this
    mode so operators can ratchet down ALL raw DELETEs and supply an
    explicit allowlist for the small handful that are intentionally
    raw (e.g. derived-data tables like ``document_chunks``).
    """
    rel = _rel(file)
    if apply_default_allowlist and rel in PATTERN1_ALLOWLIST:
        return []
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_text_call(node)):
            continue
        body = _string_literal(node)
        if body is None:
            continue
        match = _DELETE_FROM_RE.match(body)
        if match is None:
            continue
        table = match.group(1).lower()
        if not match_all_tables and table not in soft_tables:
            continue
        if match_all_tables and table in soft_tables:
            reason = (
                f"raw DELETE on soft-delete table {table!r} -- use soft_delete_cascade() instead"
            )
        elif match_all_tables:
            reason = f"raw DELETE on table {table!r}"
        else:
            reason = (
                f"raw DELETE on soft-delete table {table!r} -- use soft_delete_cascade() instead"
            )
        out.append(Violation(file=rel, line=node.lineno, reason=reason))
    return out


# --- Pattern 2: session.execute(sa.delete(M)) / session.execute(sa.update(M)) --


def _execute_call_arg(node: ast.Call) -> ast.Call | None:
    """If ``node`` is ``X.execute(<Call>, ...)``, return the inner Call."""
    if not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "execute":
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Call):
        return first
    return None


def _bulk_dml_kind(call: ast.Call) -> str | None:
    """Return ``'delete'``/``'update'`` if ``call`` is ``sa.delete(M)`` etc."""
    func = call.func
    if isinstance(func, ast.Name):
        if func.id in {"delete", "update"}:
            return func.id
        return None
    if isinstance(func, ast.Attribute):
        if func.attr in {"delete", "update"}:
            return func.attr
        return None
    return None


def _model_name(call: ast.Call) -> str | None:
    """If ``call`` is ``something(<Name>, ...)``, return the Name's id."""
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Name):
        return first.id
    if isinstance(first, ast.Attribute):
        # e.g. ``models.Course`` -- pick the rightmost attribute.
        return first.attr
    return None


def scan_bulk_dml(
    file: Path,
    *,
    protected_tables: frozenset[str],
    class_to_table: dict[str, str],
    apply_default_allowlist: bool = True,
) -> list[Violation]:
    """Find ``session.execute(sa.delete|update(<protected-model>))`` sites."""
    rel = _rel(file)
    if apply_default_allowlist and rel in PATTERN2_ALLOWLIST:
        return []
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        inner = _execute_call_arg(node)
        if inner is None:
            continue
        kind = _bulk_dml_kind(inner)
        if kind is None:
            continue
        model = _model_name(inner)
        if model is None:
            continue
        # Resolve via the registry. If the symbol isn't a known model
        # class, skip silently (could be a local variable or unrelated
        # construct).
        tablename = class_to_table.get(model)
        if tablename is None:
            continue
        if tablename not in protected_tables:
            continue
        out.append(
            Violation(
                file=rel,
                line=node.lineno,
                reason=(
                    f"bulk Core DML ({kind}) on protected model {model!r} "
                    f"(table {tablename!r}) -- bypasses audit listener; "
                    "use per-row ORM ops or soft_delete_cascade()"
                ),
            )
        )
    return out


# --- Pattern 3: text("SELECT ... FROM <soft_delete_table>") missing filter --

_FROM_OR_JOIN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:ONLY\s+)?\"?(\w+)\"?",
    re.IGNORECASE,
)
_DELETED_AT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bdeleted_at\s+IS\s+NULL\b",
    re.IGNORECASE,
)
_STARTS_SELECT_OR_WITH_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:--[^\n]*\n\s*)*(?:SELECT|WITH)\b",
    re.IGNORECASE,
)


def _path_substring_allowlisted(rel: str) -> bool:
    return any(token in rel for token in PATTERN3_PATH_SUBSTRING_ALLOWLIST)


def scan_missing_filter(
    file: Path,
    *,
    soft_tables: frozenset[str],
    apply_default_allowlist: bool = True,
) -> list[Violation]:
    """Find ``text("SELECT ... FROM <soft_delete_table>")`` lacking ``deleted_at IS NULL``."""
    rel = _rel(file)
    if apply_default_allowlist and (rel in PATTERN3_ALLOWLIST or _path_substring_allowlisted(rel)):
        return []
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_text_call(node)):
            continue
        body = _string_literal(node)
        if body is None:
            continue
        if _STARTS_SELECT_OR_WITH_RE.match(body) is None:
            continue
        referenced = {m.group(1).lower() for m in _FROM_OR_JOIN_RE.finditer(body)}
        unfiltered_soft = referenced & soft_tables
        if not unfiltered_soft:
            continue
        if _DELETED_AT_RE.search(body):
            continue
        # Order tables for stable output.
        tables_str = ", ".join(sorted(unfiltered_soft))
        out.append(
            Violation(
                file=rel,
                line=node.lineno,
                reason=(
                    f"raw SELECT references soft-delete table(s) [{tables_str}] "
                    "without `deleted_at IS NULL` filter"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration -- single entry point for both pytest + CLI
# ---------------------------------------------------------------------------


_KIND_RAW_DELETE: Final[str] = "raw_delete"
_KIND_BULK_DML: Final[str] = "bulk_dml"
_KIND_MISSING_FILTER: Final[str] = "missing_filter"
ALL_KINDS: Final[tuple[str, ...]] = (_KIND_RAW_DELETE, _KIND_BULK_DML, _KIND_MISSING_FILTER)


def scan(
    kind: str,
    *,
    root: Path = ABRIDGEAI_ROOT,
    extra_path_allowlist: frozenset[str] = frozenset(),
    apply_default_allowlist: bool = True,
) -> list[Violation]:
    """Scan ``root`` for the named pattern. Returns sorted violations.

    ``apply_default_allowlist=False`` is for the CLI gate: the caller
    supplies the entire allowlist explicitly via ``extra_path_allowlist``
    so the count reported is the raw underlying violation count and the
    user can ratchet ``--max-allowed`` down without surprise. When the
    default allowlist is disabled the ``raw_delete`` scan also widens
    its match to cover every raw ``text()`` DELETE (regardless of
    target table) so the CLI can ratchet both kinds of legacy raws.
    """
    tables = get_model_tables()
    files = list(python_files(root))

    out: list[Violation] = []
    if kind == _KIND_RAW_DELETE:
        for f in files:
            if _rel(f) in extra_path_allowlist:
                continue
            out.extend(
                scan_raw_delete(
                    f,
                    soft_tables=tables.soft_delete,
                    apply_default_allowlist=apply_default_allowlist,
                    match_all_tables=not apply_default_allowlist,
                )
            )
    elif kind == _KIND_BULK_DML:
        class_to_table = get_model_class_to_tablename()
        for f in files:
            if _rel(f) in extra_path_allowlist:
                continue
            out.extend(
                scan_bulk_dml(
                    f,
                    protected_tables=tables.protected_union,
                    class_to_table=class_to_table,
                    apply_default_allowlist=apply_default_allowlist,
                )
            )
    elif kind == _KIND_MISSING_FILTER:
        for f in files:
            if _rel(f) in extra_path_allowlist:
                continue
            out.extend(
                scan_missing_filter(
                    f,
                    soft_tables=tables.soft_delete,
                    apply_default_allowlist=apply_default_allowlist,
                )
            )
    else:
        raise ValueError(f"unknown kind: {kind!r}; expected one of {ALL_KINDS}")
    out.sort(key=lambda v: (v.file, v.line))
    return out


__all__ = [
    "ABRIDGEAI_ROOT",
    "ALL_KINDS",
    "ModelTables",
    "PATTERN1_ALLOWLIST",
    "PATTERN2_ALLOWLIST",
    "PATTERN3_ALLOWLIST",
    "PATTERN3_PATH_SUBSTRING_ALLOWLIST",
    "REPO_ROOT",
    "Violation",
    "get_model_class_to_tablename",
    "get_model_tables",
    "python_files",
    "scan",
    "scan_bulk_dml",
    "scan_missing_filter",
    "scan_raw_delete",
]
