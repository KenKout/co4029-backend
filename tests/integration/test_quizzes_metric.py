"""Phase 5 god-file kill metric for the quizzes feature (T5.15).

Two structural checks that close Phase 5:

1. **No god files** — the legacy
   ``backend/app/ai/haystack/pipelines/quiz_generation.py`` was 1297 LOC.
   Phase 5 split it into stage subfolders + 3 pipelines + services
   + routers. This test asserts no single file under
   ``abridgeai/features/quizzes/`` exceeds 800 LOC (the relaxed cap
   per the orchestrator's "correctness over brevity" directive — see
   notepad). The headline metric is that the **largest survivor** is
   well under half the legacy LOC.

2. **Prompts live in ``.j2`` files** — Jinja2 template files under
   ``ai/stages/**/prompts/`` are present and discoverable. As a
   companion negative check, no Python file under ``ai/stages/`` may
   carry a triple-quoted block that looks like an LLM prompt
   (heuristic: contains both "You are" AND "JSON" substrings). Inline
   LLM prompts in ``.py`` would defeat the migration's promise that
   prompt review is doable without reading code.
"""

from __future__ import annotations

import re
from pathlib import Path

LOC_CAP = 800
HEURISTIC_PROMPT_TOKENS = ("You are", "JSON")
QUIZZES_FEATURE = Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "quizzes"
QUIZ_STAGES = QUIZZES_FEATURE / "ai" / "stages"


def _python_files() -> list[Path]:
    return [path for path in QUIZZES_FEATURE.rglob("*.py") if "__pycache__" not in path.parts]


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_no_god_file_under_quizzes_feature() -> None:
    files = _python_files()
    assert files, f"no python files discovered under {QUIZZES_FEATURE}"

    sized = sorted(((_line_count(p), p) for p in files), reverse=True)
    over_cap = [(loc, p) for loc, p in sized if loc > LOC_CAP]

    if over_cap:
        breakdown = "\n".join(
            f"  {loc:>5d}  {path.relative_to(QUIZZES_FEATURE)}" for loc, path in sized
        )
        offenders = "\n".join(
            f"  {loc:>5d}  {path.relative_to(QUIZZES_FEATURE)}" for loc, path in over_cap
        )
        raise AssertionError(
            f"{len(over_cap)} file(s) exceed the {LOC_CAP} LOC cap:\n"
            f"{offenders}\n\nFull descending breakdown:\n{breakdown}"
        )

    largest_loc, _ = sized[0]
    assert largest_loc < LOC_CAP, f"max LOC must be < {LOC_CAP} (saw {largest_loc})"


def test_jinja_prompts_live_in_j2_files() -> None:
    j2_files = sorted(QUIZ_STAGES.rglob("*.j2"))
    assert j2_files, (
        f"expected at least one .j2 prompt under {QUIZ_STAGES}; "
        "Phase 5 promised LLM prompts live in Jinja2 templates"
    )

    expected_stages = {"generation", "ideation", "validation"}
    found_stages = {p.parent.parent.name for p in j2_files} | {
        p.parent.parent.parent.name for p in j2_files
    }
    missing = expected_stages - found_stages
    assert not missing, f"expected .j2 prompts for stages {expected_stages}; missing for {missing}"


_TRIPLE_BLOCK = re.compile(r'"""(.*?)"""', re.DOTALL)


def test_no_inline_llm_prompts_in_stage_python() -> None:
    offenders: list[tuple[Path, str]] = []
    for py_file in QUIZ_STAGES.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8")
        for match in _TRIPLE_BLOCK.finditer(text):
            block = match.group(1)
            if all(token in block for token in HEURISTIC_PROMPT_TOKENS):
                offenders.append((py_file, block[:200]))

    if offenders:
        formatted = "\n".join(
            f"{path.relative_to(QUIZZES_FEATURE)}: {snippet!r}" for path, snippet in offenders
        )
        raise AssertionError(
            "Inline LLM-prompt-shaped triple-quoted blocks found in stage "
            f"Python — move them to .j2 files:\n{formatted}"
        )


__all__: list[str] = []
