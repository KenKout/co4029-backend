"""FIX-CRIT-3 regression: divergent ``effective_permissions.sql`` must NOT
exist in ``backend-new/``.

The legacy ``backend/app/queries/sql/permissions/effective_permissions.sql``
was incomplete (no direct grants, no active-window filter). The canonical
replacement lives in
:mod:`abridgeai.features.access_control.queries.permissions`. These tests
lock that perimeter so the divergent SQL file can never silently re-appear.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_divergent_sql_file() -> None:
    """If anyone re-introduces ``effective_permissions.sql``, fail loudly."""
    matches = list(REPO_ROOT.rglob("effective_permissions.sql"))
    assert matches == [], (
        f"effective_permissions.sql found at {matches}. "
        "This file was the divergent legacy query (FIX-CRIT-3). "
        "Use abridgeai.features.access_control.queries.load_user_permissions "
        "instead."
    )


def test_canonical_module_docstring_documents_rationale() -> None:
    """Module docstring must explain why the legacy SQL is not ported."""
    from abridgeai.features.access_control.queries import permissions as mod

    doc = (mod.__doc__ or "").lower()
    assert "effective_permissions.sql" in doc, (
        "Canonical module docstring must reference the legacy file by name."
    )
    assert "not ported" in doc, (
        "Canonical module docstring must state that the legacy file is not ported."
    )
    assert "canonical" in doc, "Canonical module docstring must self-identify as canonical."
