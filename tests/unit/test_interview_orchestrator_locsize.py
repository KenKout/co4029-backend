"""No-god-file LOC guard for the interview orchestrator directory.

Mirrors the generation-stage guard (test_interview_generation_stage.py):
enforce the feature-wide 800-LOC "no god files" ceiling, plus tighter soft
budgets on the files the v2 decomposition (Slice 7) keeps lean. adaptive.py
starts at 689 LOC — this guard is RED until Task 7.1 splits it into
turn_perception / turn_state helpers.
"""

from __future__ import annotations

from pathlib import Path


def test_no_god_file_in_orchestrator() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "orchestrator"
    assert target.is_dir()
    hard_cap = 800  # feature-wide "no god files" ceiling
    soft_budget = {
        # adaptive.py is the orchestration seam: it threads each v2 feature's
        # flag + small wiring block through run_adaptive_turn, so it grows a
        # little per slice. Pure logic lives in sibling modules (phases.py,
        # turn_state.py, difficulty.py, ...), not here. 500 keeps it lean with
        # headroom, well under the 800 feature-wide hard cap.
        "adaptive.py": 500,
        # decision.py is the deterministic policy core: every v2 slice that adds
        # a decision rule (depth probe, rich-closing sub-state machine) grows it.
        # Its helpers build InterviewerDecision objects, so they can't extract to
        # a pure sibling the way phases.py does without a circular import. 620
        # keeps it honest with headroom, well under the 800 feature-wide cap.
        "decision.py": 620,
    }
    offenders = []
    for path in target.rglob("*.py"):
        with path.open() as fh:
            loc = sum(1 for _ in fh)
        cap = soft_budget.get(path.name, hard_cap)
        if loc > cap:
            offenders.append(f"{path.name}: {loc} > {cap}")
    assert not offenders, "orchestrator god-files: " + "; ".join(offenders)
