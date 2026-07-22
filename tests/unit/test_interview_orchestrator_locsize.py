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
        # turn_state.py, difficulty.py, ...), not here. Each v2 slice adds a
        # flag param + a small wiring block (and the realism cluster added the
        # _is_rambling / _comms_polish_signals seam helpers), so it creeps up a
        # little per slice. 580 keeps it lean with headroom, well under the 800
        # feature-wide hard cap.
        "adaptive.py": 580,
        # decision.py is the deterministic policy core: every v2 slice that adds
        # a decision rule (depth probe, rich-closing sub-state machine, the
        # realism cluster — self-correction / confident-but-wrong / rambling
        # redirect) grows it. Its helpers build InterviewerDecision objects, so
        # they can't extract to a pure sibling the way phases.py does without a
        # circular import. Rule bodies are also extracted into same-file helpers
        # (_depth_probe / _confident_wrong_challenge / _rambling_redirect /
        # _advance_reason) to keep decide_next_action under the cyclomatic-
        # complexity cap, which trades a little length for readability. When it
        # hit the 800 ceiling (Slice 19), the three pure enums (action /
        # acknowledgement / reason) were extracted to decision_types.py and
        # re-exported, dropping it back to ~730. 760 keeps headroom under 800.
        "decision.py": 760,
    }
    offenders = []
    for path in target.rglob("*.py"):
        with path.open() as fh:
            loc = sum(1 for _ in fh)
        cap = soft_budget.get(path.name, hard_cap)
        if loc > cap:
            offenders.append(f"{path.name}: {loc} > {cap}")
    assert not offenders, "orchestrator god-files: " + "; ".join(offenders)
