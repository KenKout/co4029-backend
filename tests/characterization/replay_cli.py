from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .replay import replay


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.characterization.replay_cli",
        description="Replay an endpoint and compare with the captured snapshot under its parity contract.",
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("old", "new"),
        help="Backend to replay AGAINST (the one being verified).",
    )
    parser.add_argument(
        "--captured",
        choices=("old", "new"),
        default="old",
        help="Which saved snapshot to compare against (default: old).",
    )
    parser.add_argument("--method", default="GET")
    return parser


async def _run(args: argparse.Namespace) -> int:
    result = await replay(
        args.endpoint,
        backend=args.backend,
        method=args.method,
        captured_backend=args.captured,
    )
    print(
        json.dumps(
            {
                "endpoint": args.endpoint,
                "passed": result.passed,
                "divergences": result.divergences,
            },
            indent=2,
        )
    )
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
