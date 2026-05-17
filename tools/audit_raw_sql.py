"""CLI gate for the three raw-SQL audit-bypass patterns (T4).

Reuses the scan engine from ``tests.lint._model_introspection`` so the
CLI tool and the pytest lint check enforce IDENTICAL rules. Used by the
Wave 6 success-criteria gate:

  python tools/audit_raw_sql.py \\
      --kind raw_delete \\
      --max-allowed 1 \\
      --allowlist abridgeai/features/materials/services/authoring/_storage.py

Exits ``0`` when ``violations <= max-allowed``, ``1`` otherwise. Prints
violations to stderr (one per line, ``<file>:<line>: <reason>``) and a
summary line to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.lint._model_introspection import (  # noqa: E402
    ABRIDGEAI_ROOT,
    ALL_KINDS,
    scan,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_raw_sql",
        description=(
            "Static AST scan for raw-SQL audit-bypass anti-patterns. "
            "Exits 0 when violations <= --max-allowed."
        ),
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=list(ALL_KINDS),
        help="Anti-pattern to scan for.",
    )
    parser.add_argument(
        "--max-allowed",
        type=int,
        default=0,
        help="Maximum allowed violation count (default: 0).",
    )
    parser.add_argument(
        "--allowlist",
        default="",
        help=(
            "Colon-separated repo-relative paths to additionally allowlist "
            "for this run (additive to defaults baked into the scanner)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ABRIDGEAI_ROOT,
        help=f"Directory to scan (default: {ABRIDGEAI_ROOT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    extra = frozenset(p for p in args.allowlist.split(":") if p)
    violations = scan(
        args.kind,
        root=args.root,
        extra_path_allowlist=extra,
        apply_default_allowlist=False,
    )

    for v in violations:
        print(v.render(), file=sys.stderr)

    count = len(violations)
    print(
        f"audit_raw_sql kind={args.kind} count={count} max-allowed={args.max_allowed}",
    )
    return 0 if count <= args.max_allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
