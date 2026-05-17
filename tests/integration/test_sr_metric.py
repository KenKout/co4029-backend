"""Phase 7.5 god-file kill metric for the spaced_repetition feature (T7.5.13).

Asserts no single file under ``abridgeai/features/spaced_repetition/``
exceeds the 800 LOC cap (relaxed from the plan body's 300 per the
orchestrator's "correctness over brevity" directive — see
``test_quizzes_metric.py`` for the same threshold). Mirrors the Phase 5
quizzes / Phase 6 interviews metric tests so future SR refactors don't
silently regrow into a god file.
"""

from __future__ import annotations

from pathlib import Path

LOC_CAP = 800
SR_FEATURE = Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "spaced_repetition"


def _python_files() -> list[Path]:
    return [path for path in SR_FEATURE.rglob("*.py") if "__pycache__" not in path.parts]


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_no_god_file_under_spaced_repetition_feature() -> None:
    files = _python_files()
    assert files, f"no python files discovered under {SR_FEATURE}"

    sized = sorted(((_line_count(p), p) for p in files), reverse=True)
    over_cap = [(loc, p) for loc, p in sized if loc > LOC_CAP]

    if over_cap:
        breakdown = "\n".join(f"  {loc:>5d}  {path.relative_to(SR_FEATURE)}" for loc, path in sized)
        offenders = "\n".join(
            f"  {loc:>5d}  {path.relative_to(SR_FEATURE)}" for loc, path in over_cap
        )
        raise AssertionError(
            f"{len(over_cap)} file(s) exceed the {LOC_CAP} LOC cap:\n"
            f"{offenders}\n\nFull descending breakdown:\n{breakdown}"
        )

    largest_loc, _ = sized[0]
    assert largest_loc < LOC_CAP, f"max LOC must be < {LOC_CAP} (saw {largest_loc})"


__all__: list[str] = []
