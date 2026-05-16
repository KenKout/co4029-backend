"""CLI: ``python -m abridgeai.access_control.validate_catalog``.

Exits 0 if ``catalog.yaml`` and ``role_seeds.yaml`` parse, validate, and
cross-validate; non-zero with a stderr error on any failure. Used by
pre-commit, CI, and local sanity checks.
"""

from __future__ import annotations

import sys

from abridgeai.access_control.permissions.loader import (
    load_catalog,
    load_role_seeds,
)


def main() -> int:
    try:
        catalog = load_catalog()
        seeds = load_role_seeds(catalog)
    except Exception as exc:
        sys.stderr.write(f"catalog validation FAILED: {exc}\n")
        return 1
    sys.stdout.write(
        f"OK \u2014 {len(catalog.permissions)} permissions, {len(seeds.roles)} roles\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
