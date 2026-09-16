"""Regression tests for the guarded formula-version column removal."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "0126_drop_formula_version.py"
)
_SPEC = importlib.util.spec_from_file_location("drop_formula_version", _MIGRATION_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load migration module")
_MIGRATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MIGRATION)


def test_upgrade_refuses_to_drop_non_v2_snapshot_history() -> None:
    bind = MagicMock()
    bind.scalar.return_value = 1

    with (
        patch.object(_MIGRATION.op, "get_bind", return_value=bind),
        patch.object(_MIGRATION.op, "drop_column") as drop_column,
        pytest.raises(RuntimeError, match="formula other than 2"),
    ):
        _MIGRATION.upgrade()

    drop_column.assert_not_called()
