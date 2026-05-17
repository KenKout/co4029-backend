"""Static lint check: forbid raw-SQL audit/soft-delete bypasses (T4).

Three pytest functions assert that the production codebase under
``abridgeai/`` is free of three anti-patterns that would silently
bypass the audit / hard-delete-guard / soft-delete-loader safety net.
A fourth test re-runs the same scanners against the fixtures directory
and asserts each fixture IS detected (positive-case verification),
which keeps the scanner honest.

The three patterns and their detection logic live in
``tests/lint/_model_introspection.py``; this file is just the pytest
host plus a runtime budget assertion.
"""

from __future__ import annotations

import time
from pathlib import Path

from tests.lint._model_introspection import (
    ABRIDGEAI_ROOT,
    ALL_KINDS,
    REPO_ROOT,
    Violation,
    get_model_class_to_tablename,
    get_model_tables,
    scan,
    scan_bulk_dml,
    scan_missing_filter,
    scan_raw_delete,
)

LINT_RUNTIME_BUDGET_S = 30.0
FIXTURES_DIR = Path(__file__).resolve().parent / "_fixtures"


def _format(violations: list[Violation]) -> str:
    return "\n".join(v.render() for v in violations) or "(none)"


def test_no_raw_delete_on_soft_delete_tables() -> None:
    """Pattern 1: ``text('DELETE FROM <soft_delete_table>')`` is forbidden."""
    violations = scan("raw_delete", root=ABRIDGEAI_ROOT)
    assert not violations, (
        f"{len(violations)} raw DELETE on soft-delete table(s) found:\n"
        f"{_format(violations)}\n\n"
        "Use soft_delete_cascade() or per-row ORM ops; if the target "
        "is genuinely derived data (no audit trail required), add the "
        "file to PATTERN1_ALLOWLIST in tests/lint/_model_introspection.py "
        "with a code comment explaining why."
    )


def test_no_bulk_core_dml_on_protected_models() -> None:
    """Pattern 2: ``session.execute(sa.delete|update(M))`` on audited/soft-deleted M."""
    violations = scan("bulk_dml", root=ABRIDGEAI_ROOT)
    assert not violations, (
        f"{len(violations)} bulk Core DML on protected model(s) found:\n"
        f"{_format(violations)}\n\n"
        "Bulk DML LOOKS like ORM but bypasses the unit-of-work; the "
        "audit listener never fires and `deleted_at` is never stamped. "
        "Use per-row ORM ops or soft_delete_cascade() instead. "
        "PATTERN2_ALLOWLIST is intentionally empty -- zero tolerance."
    )


def test_no_raw_select_missing_deleted_at_filter() -> None:
    """Pattern 3: ``text('SELECT ... FROM <soft_delete_table>')`` must filter ``deleted_at``."""
    violations = scan("missing_filter", root=ABRIDGEAI_ROOT)
    assert not violations, (
        f"{len(violations)} raw SELECT(s) missing `deleted_at IS NULL`:\n"
        f"{_format(violations)}\n\n"
        "Raw SQL bypasses the ORM soft-delete loader-criteria. "
        "Add `<table>.deleted_at IS NULL` to the WHERE clause, or "
        "extend PATTERN3_ALLOWLIST in tests/lint/_model_introspection.py "
        "if the query legitimately must see soft-deleted rows."
    )


def test_fixtures_are_detected() -> None:
    """Positive-case verification: every fixture must be caught by its scanner."""
    tables = get_model_tables()
    class_to_table = get_model_class_to_tablename()

    raw_delete_fixture = FIXTURES_DIR / "bad_raw_delete.py"
    bulk_dml_fixture = FIXTURES_DIR / "bad_bulk_dml.py"
    missing_filter_fixture = FIXTURES_DIR / "bad_missing_filter.py"

    raw_delete_hits = scan_raw_delete(raw_delete_fixture, soft_tables=tables.soft_delete)
    assert len(raw_delete_hits) == 1, (
        f"expected 1 raw_delete violation in fixture; got {raw_delete_hits!r}"
    )
    assert "courses" in raw_delete_hits[0].reason

    bulk_dml_hits = scan_bulk_dml(
        bulk_dml_fixture,
        protected_tables=tables.protected_union,
        class_to_table=class_to_table,
    )
    assert len(bulk_dml_hits) == 1, (
        f"expected 1 bulk_dml violation in fixture; got {bulk_dml_hits!r}"
    )
    assert "Course" in bulk_dml_hits[0].reason

    missing_filter_hits = scan_missing_filter(
        missing_filter_fixture, soft_tables=tables.soft_delete
    )
    assert len(missing_filter_hits) == 1, (
        f"expected 1 missing_filter violation in fixture; got {missing_filter_hits!r}"
    )
    assert "courses" in missing_filter_hits[0].reason


def test_allowlist_paths_pass_clean() -> None:
    """The hand-curated allowlist paths must produce zero violations under their own scanner."""
    violations = scan("raw_delete", root=ABRIDGEAI_ROOT)
    storage = "abridgeai/features/materials/services/authoring/_storage.py"
    assert all(v.file != storage for v in violations), (
        f"{storage} should be allowlisted for raw_delete; saw violations: "
        f"{[v.render() for v in violations if v.file == storage]}"
    )
    bootstrap_paths = {
        "abridgeai/core/security/__init__.py",
        "abridgeai/core/observability/audit_log.py",
        "abridgeai/ai/llm/embeddings.py",
        "abridgeai/api/healthz.py",
    }
    select_violations = scan("missing_filter", root=ABRIDGEAI_ROOT)
    leaked = [v for v in select_violations if v.file in bootstrap_paths]
    assert not leaked, (
        f"bootstrap allowlist is leaking missing_filter violations: {[v.render() for v in leaked]}"
    )


def test_lint_runtime_within_budget() -> None:
    """The full three-pattern scan must complete in < 30s on the codebase."""
    start = time.perf_counter()
    for kind in ALL_KINDS:
        scan(kind, root=ABRIDGEAI_ROOT)
    elapsed = time.perf_counter() - start
    assert elapsed < LINT_RUNTIME_BUDGET_S, (
        f"lint scan took {elapsed:.2f}s (budget {LINT_RUNTIME_BUDGET_S}s)"
    )
    print(f"\n[lint] scan elapsed: {elapsed:.2f}s")


def test_repo_geometry_resolves_correctly() -> None:
    """Sanity check: the path constants point where we think they do."""
    assert (REPO_ROOT / "pyproject.toml").exists(), (
        f"REPO_ROOT misresolved -- expected backend-new/, got {REPO_ROOT}"
    )
    assert ABRIDGEAI_ROOT.is_dir()
    assert FIXTURES_DIR.is_dir()


__all__: list[str] = []
