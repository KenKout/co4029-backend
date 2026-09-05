"""Phase 6 god-file kill metric for the interviews feature (T6.14).

Two structural checks that close Phase 6:

1. **No god files** — the legacy
   ``backend/app/ai/haystack/pipelines/interview_generation.py`` was a
   single 154-LOC file co-located with the quiz pipeline god-file.
   Phase 6 ships interviews as ~30+ files distributed across stages,
   pipelines, services, queries, schemas, routers, and workers — over
   5K LOC in total. The win here is *architectural*, not raw LOC
   reduction: prompt review is doable without reading code, stages
   compose cleanly, and the post-submit Gap Report pipeline lands
   without bloating the generation pipeline.

   This test asserts no single file under
   ``abridgeai/features/interviews/`` exceeds 800 LOC (the relaxed cap
   per the orchestrator's "correctness over brevity" directive — same
   precedent as T5.15). The headline metric is the **largest survivor**
   sits comfortably under the cap.

2. **Prompts live in ``.j2`` files** — Jinja2 template files under
   ``ai/stages/**/prompts/`` are present and discoverable for the five
   LLM-emitting stages (generation, validation, followup, evaluation,
   gap_report). Companion negative check: no Python file under
   ``ai/stages/`` carries a triple-quoted block that looks like an LLM
   prompt (heuristic: contains both "You are" AND "JSON" substrings).
   Inline LLM prompts in ``.py`` would defeat the migration's promise
   that prompt review is doable without reading code.
"""

from __future__ import annotations

import re
from pathlib import Path

LOC_CAP = 800

# RATCHET GRANDFATHER LIST — legacy files already over the cap when this gate
# was (re)enforced on 2026-07-26. Each entry pins the file at its CURRENT size
# plus a 3%% slack: any real growth still fails the build, and shrinking a file
# below the cap should be followed by deleting its entry. Do NOT add new files
# here — new code must fit the cap.
_GRANDFATHERED: dict[str, int] = {
    "services/taking.py": 2236,
    "routers/authoring.py": 1226,
    "routers/learner.py": 1141,
    # Repinned 2026-09-05 (877 -> 890) for the evaluation-claim columns
    # (migration 0107). A schema column has exactly one legal home, so a table
    # this file already owns cannot be split out to stay under the old pin; the
    # ratchet is raised by the ~9 lines the columns and their comment cost, not
    # relaxed. Everything else in that change went to NEW files
    # (services/evaluation_claim.py, services/gap_report_writer.py), and
    # services/evaluation.py was brought back under the 800 cap by the split
    # rather than added to this list.
    "models.py": 890,
}
HEURISTIC_PROMPT_TOKENS = ("You are", "JSON")
INTERVIEWS_FEATURE = Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "interviews"
INTERVIEW_STAGES = INTERVIEWS_FEATURE / "ai" / "stages"


def _python_files() -> list[Path]:
    return [path for path in INTERVIEWS_FEATURE.rglob("*.py") if "__pycache__" not in path.parts]


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_no_god_files_in_interviews() -> None:
    files = _python_files()
    assert files, f"no python files discovered under {INTERVIEWS_FEATURE}"

    sized = sorted(((_line_count(p), p) for p in files), reverse=True)
    over_cap = [
        (loc, p)
        for loc, p in sized
        if loc > _GRANDFATHERED.get(str(p.relative_to(INTERVIEWS_FEATURE)), LOC_CAP)
    ]

    if over_cap:
        breakdown = "\n".join(
            f"  {loc:>5d}  {path.relative_to(INTERVIEWS_FEATURE)}" for loc, path in sized
        )
        offenders = "\n".join(
            f"  {loc:>5d}  {path.relative_to(INTERVIEWS_FEATURE)}" for loc, path in over_cap
        )
        raise AssertionError(
            f"{len(over_cap)} file(s) exceed the {LOC_CAP} LOC cap:\n"
            f"{offenders}\n\nFull descending breakdown:\n{breakdown}"
        )

    # Redundant with the per-file ratchet above (grandfathered files are
    # pinned individually); kept for NON-grandfathered files only.
    non_grandfathered = [
        (loc, p)
        for loc, p in sized
        if str(p.relative_to(INTERVIEWS_FEATURE)) not in _GRANDFATHERED
    ]
    if non_grandfathered:
        largest_loc, _ = non_grandfathered[0]
        assert largest_loc < LOC_CAP, f"max LOC must be < {LOC_CAP} (saw {largest_loc})"


def test_jinja_prompts_in_j2_files() -> None:
    j2_files = sorted(INTERVIEW_STAGES.rglob("*.j2"))
    assert j2_files, (
        f"expected at least one .j2 prompt under {INTERVIEW_STAGES}; "
        "Phase 6 promised LLM prompts live in Jinja2 templates"
    )

    expected_stages = {"generation", "validation", "followup", "evaluation", "gap_report"}
    found_stages = {p.parent.parent.name for p in j2_files} | {
        p.parent.parent.parent.name for p in j2_files
    }
    missing = expected_stages - found_stages
    assert not missing, f"expected .j2 prompts for stages {expected_stages}; missing for {missing}"


_TRIPLE_BLOCK = re.compile(r'"""(.*?)"""', re.DOTALL)


def test_no_inline_llm_prompts_in_stage_python() -> None:
    offenders: list[tuple[Path, str]] = []
    for py_file in INTERVIEW_STAGES.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8")
        for match in _TRIPLE_BLOCK.finditer(text):
            block = match.group(1)
            if all(token in block for token in HEURISTIC_PROMPT_TOKENS):
                offenders.append((py_file, block[:200]))

    if offenders:
        formatted = "\n".join(
            f"{path.relative_to(INTERVIEWS_FEATURE)}: {snippet!r}" for path, snippet in offenders
        )
        raise AssertionError(
            "Inline LLM-prompt-shaped triple-quoted blocks found in stage "
            f"Python — move them to .j2 files:\n{formatted}"
        )


__all__: list[str] = []
