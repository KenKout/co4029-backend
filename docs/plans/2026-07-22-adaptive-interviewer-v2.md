# Adaptive Interviewer v2 — Real-Life Interview Coverage — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Upgrade the adaptive interview orchestrator so a live session behaves like a competent human interviewer — easing in, escalating depth on strong answers, remembering prior claims across turns, reading candidate affect, laddering hints, calibrating difficulty per outcome, and closing gracefully.

**Architecture:** Preserve the existing split — pure deterministic policy modules (`decision.py`, `selection.py`, `difficulty.py`, `coverage.py`, `phases.py`) fed by LLM perception (`intent`, `analysis`) — with `run_adaptive_turn` as the single orchestration seam. Every new behaviour is flag-gated behind a new `ADAPTIVE_INTERVIEWER_V2_ENABLED` master (default OFF), rides the existing staged-rollout runbook, and falls back to the current v1 adaptive path (which itself falls back to legacy sequential). No change ever binds the independent post-session evaluator.

**Tech Stack:** Python 3.14 / FastAPI / SQLAlchemy async / Pydantic Settings / Jinja2 prompt templates / pytest. Verify with **pyright + ruff + pytest** (mypy 1.20.2 crashes on this Py3.14 env — do NOT run mypy). Frontend i18n: `en.json` + `vi.json` (bilingual, both required for every user-facing string).

---

## Ground rules (read before starting any task)

1. **Flag gate.** All v2 behaviour is gated by a new `adaptive_interviewer_v2_enabled` (master, default `False`) plus per-slice sub-flags (default `False`). When `v2` is OFF, `run_adaptive_turn` must produce byte-for-byte the current v1 result. This is the one-flip kill switch and the guarantee that existing traffic is untouched.
2. **Determinism where it matters.** Control-flow decisions (advance / probe / close / phase transition) stay in pure, unit-testable policy modules with NO DB and NO LLM. The LLM only feeds perception (intent, analysis, affect) and reshapes phrasing — it never decides control flow. This preserves the brief's non-goal ("do not depend entirely on an LLM for deterministic state transitions").
3. **File-size cap.** The interviews feature enforces an 800-LOC/file "no god files" cap (see Task 0.3 — we ADD an explicit test for the orchestrator dir). `adaptive.py` is already 689 lines; **Slice 7 Task 7.1 decomposes it first** so later slices don't blow the cap. Keep every new/edited orchestrator file ≤ 400 LOC where practical, hard-fail at 800.
4. **i18n.** Every new student-facing utterance template string needs BOTH `en.json` and `vi.json` keys. A missing `vi.json` key is a release blocker.
5. **Idempotency / one-version-owner / savepoint invariants are sacred.** New state fields are added to `InterviewRuntimeStateData` with tolerant `from_dict` defaults (bump `STATE_SCHEMA_VERSION`). `run_adaptive_turn` still calls `state_repo.save` EXACTLY once. Never add a second save or a nested commit.
6. **Migrations.** No schema/table change is needed — runtime state is JSONB and tolerant. Only `STATE_SCHEMA_VERSION` bumps. If any slice ever needs a real column, revision ids MUST be ≤32 chars (varchar(32) `alembic_version` cap).
7. **TDD.** Every code task: write failing test → run to confirm RED → minimal implementation → run to confirm GREEN → commit. Commit directly to `master` (per user standing instruction: no feature branches, no merges). Push at each slice boundary.
8. **Prove clean tree first.** Before Slice 7, run the full interview test suite on the untouched tree and record the baseline (Task 0.1). Never blame a pre-existing failure on your diff.

Test command (single source of truth used throughout):

```bash
cd /root/co4029/backend && . .venv/bin/activate
python -m pytest tests/unit tests/integration -k "interview or adaptive" -q
```

Lint/type command:

```bash
cd /root/co4029/backend && . .venv/bin/activate && ruff check abridgeai/features/interviews && pyright abridgeai/features/interviews
```

---

## Slice 0 — Baseline, flags, and the LOC guard

### Task 0.1: Capture the clean-tree baseline

**Objective:** Prove which interview tests pass on the untouched tree so no later failure is misattributed.

**Step 1: Run the suite**

```bash
cd /root/co4029/backend && . .venv/bin/activate
python -m pytest tests/unit tests/integration -k "interview or adaptive" -q | tee /tmp/adaptive_v2_baseline.txt
```

Expected: record the pass/fail/skip counts. This file is the reference; any regression must be traceable to a task.

**Step 2: No commit** (read-only baseline).

---

### Task 0.2: Add the v2 flag matrix to Settings

**Objective:** Introduce the v2 master switch + per-slice sub-flags, defaulting OFF, mirroring the existing v1 flag conventions.

**Files:**
- Modify: `abridgeai/core/config.py` (after line 248, alongside the existing adaptive flags)
- Test: `tests/unit/test_adaptive_mode_flags.py`

**Step 1: Write failing test**

```python
# tests/unit/test_adaptive_mode_flags.py  (add to existing file)
def test_v2_flags_default_off(base_settings):
    s = base_settings
    assert s.adaptive_interviewer_v2_enabled is False
    assert s.adaptive_v2_phases_enabled is False
    assert s.adaptive_v2_depth_probe_enabled is False
    assert s.adaptive_v2_cross_turn_enabled is False
    assert s.adaptive_v2_affect_enabled is False
    assert s.adaptive_v2_hint_ladder_enabled is False
    assert s.adaptive_v2_per_outcome_difficulty_enabled is False
    assert s.adaptive_v2_rich_closing_enabled is False


def test_v2_resolver_requires_master_and_v1(base_settings):
    s = base_settings.model_copy(update={
        "adaptive_interviewer_enabled": True,
        "adaptive_interviewer_text_enabled": True,
        "adaptive_interviewer_v2_enabled": True,
        "adaptive_v2_phases_enabled": True,
    })
    assert s.adaptive_v2_feature_enabled("text", "phases") is True
    # v2 master off → every sub-feature off even if sub-flag on
    s2 = s.model_copy(update={"adaptive_interviewer_v2_enabled": False})
    assert s2.adaptive_v2_feature_enabled("text", "phases") is False
    # v1 mode gate off → v2 cannot run for that mode
    s3 = s.model_copy(update={"adaptive_interviewer_text_enabled": False})
    assert s3.adaptive_v2_feature_enabled("text", "phases") is False
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_adaptive_mode_flags.py -k v2 -q`
Expected: FAIL — attributes/method missing.

**Step 3: Implement in `config.py`** (insert after the `adaptive_interviewer_rollout_percent` field, before the resolver methods)

```python
    # ── Adaptive Interviewer v2 (real-life coverage upgrade) ─────────────────
    # v2 layers richer interviewer behaviour on top of the v1 adaptive path.
    # MASTER v2 switch, default OFF: when OFF, run_adaptive_turn produces the
    # exact v1 result. Requires the v1 master + per-mode gate to already be on
    # (v2 cannot run where v1 does not). Each sub-feature is independently
    # gated and defaults OFF so it can shadow/canary one behaviour at a time.
    adaptive_interviewer_v2_enabled: bool = False
    adaptive_v2_phases_enabled: bool = False
    adaptive_v2_depth_probe_enabled: bool = False
    adaptive_v2_cross_turn_enabled: bool = False
    adaptive_v2_affect_enabled: bool = False
    adaptive_v2_hint_ladder_enabled: bool = False
    adaptive_v2_per_outcome_difficulty_enabled: bool = False
    adaptive_v2_rich_closing_enabled: bool = False
```

And add the resolver method next to `adaptive_enabled_for_mode`:

```python
    _V2_SUBFLAGS: ClassVar[dict[str, str]] = {
        "phases": "adaptive_v2_phases_enabled",
        "depth_probe": "adaptive_v2_depth_probe_enabled",
        "cross_turn": "adaptive_v2_cross_turn_enabled",
        "affect": "adaptive_v2_affect_enabled",
        "hint_ladder": "adaptive_v2_hint_ladder_enabled",
        "per_outcome_difficulty": "adaptive_v2_per_outcome_difficulty_enabled",
        "rich_closing": "adaptive_v2_rich_closing_enabled",
    }

    def adaptive_v2_feature_enabled(self, input_mode: str, feature: str) -> bool:
        """A v2 sub-feature runs only when: v1 mode gate ON, v2 master ON,
        AND the named sub-flag ON. Unknown feature → False (fail closed)."""
        if not self.adaptive_enabled_for_mode(input_mode):
            return False
        if not self.adaptive_interviewer_v2_enabled:
            return False
        attr = self._V2_SUBFLAGS.get(feature)
        return bool(attr and getattr(self, attr, False))
```

Add `from typing import ClassVar` to the imports if not present.

**Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_adaptive_mode_flags.py -k v2 -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add abridgeai/core/config.py tests/unit/test_adaptive_mode_flags.py
git commit -m "feat(interviews): add adaptive v2 flag matrix + resolver (default off)"
```

---

### Task 0.3: Add the orchestrator no-god-file guard

**Objective:** Add an explicit LOC-cap test for the orchestrator directory so the decomposition in Slice 7 is enforced by CI, matching the existing generation-stage guard pattern.

**Files:**
- Test: `tests/unit/test_interview_orchestrator_locsize.py` (new)

**Step 1: Write failing test**

```python
# tests/unit/test_interview_orchestrator_locsize.py
from pathlib import Path


def test_no_god_file_in_orchestrator() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "orchestrator"
    assert target.is_dir()
    hard_cap = 800  # feature-wide "no god files" ceiling
    soft_budget = {
        # decomposition targets (Slice 7); keep these lean
        "adaptive.py": 400,
        "decision.py": 550,
    }
    offenders = []
    for path in target.rglob("*.py"):
        with path.open() as fh:
            loc = sum(1 for _ in fh)
        cap = soft_budget.get(path.name, hard_cap)
        if loc > cap:
            offenders.append(f"{path.name}: {loc} > {cap}")
    assert not offenders, "orchestrator god-files: " + "; ".join(offenders)
```

**Step 2: Run to verify current state**

Run: `python -m pytest tests/unit/test_interview_orchestrator_locsize.py -q`
Expected: **FAIL** — `adaptive.py: 689 > 400`. This is the RED that Task 7.1 turns GREEN.

**Step 3: No implementation yet** — this test intentionally fails until Task 7.1. Commit the guard now so the decomposition has a target.

**Step 4: Commit**

```bash
git add tests/unit/test_interview_orchestrator_locsize.py
git commit -m "test(interviews): add orchestrator no-god-file LOC guard (RED until 7.1)"
```

---

## Slice 7 — Real phase progression (+ adaptive.py decomposition)

Wire `WARMUP` and `DEEP_PROBE` (already in the `InterviewPhase` enum, unused) into the state machine so a session flows OPENING → WARMUP → CORE → DEEP_PROBE → CLOSING, with phase biasing difficulty target and probe aggressiveness. Decompose `adaptive.py` first to make room.

### Task 7.1: Extract perception + state-application helpers out of `adaptive.py`

**Objective:** Move the pure/near-pure helpers into sibling modules so `adaptive.py` drops under 400 LOC, turning Task 0.3 GREEN, with zero behaviour change.

**Files:**
- Create: `abridgeai/features/interviews/orchestrator/turn_perception.py`
- Create: `abridgeai/features/interviews/orchestrator/turn_state.py`
- Modify: `abridgeai/features/interviews/orchestrator/adaptive.py`
- Test: existing `tests/integration/test_interview_adaptive_step.py` + `tests/unit/test_interview_orchestrator_locsize.py`

**Step 1: Confirm RED persists**

Run: `python -m pytest tests/unit/test_interview_orchestrator_locsize.py -q`
Expected: FAIL (adaptive.py too big) — the target we will fix.

**Step 2: Create `turn_state.py`** — move `_apply_state_updates`, `_sync_question_history`, `_with_replay`, `_rehydrate_replay`, `_compact_scores`, `_probe_seed_text` verbatim (they are already near-pure). Keep signatures identical; import the enums they use.

```python
"""State-application + replay helpers extracted from adaptive.py (Slice 7).

Pure-ish mutation of the loaded runtime state in memory — NO DB save (the
caller owns the single save). Extracted verbatim from run_adaptive_turn to
keep adaptive.py under the orchestrator LOC cap; behaviour is unchanged.
"""
from __future__ import annotations
# ... move the six helper functions here unchanged, with their imports ...
```

**Step 3: Create `turn_perception.py`** — move `_load_candidates`, `authoring_list_questions`, `authoring_list_outcomes`, `_persisted_question_ids`, `_time_fraction_remaining`, `_confirmation_override`, `_next_sequence`, `_no_gateway` verbatim.

**Step 4: Rewrite `adaptive.py`** to import from the two new modules. `run_adaptive_turn` keeps its exact body but calls `turn_perception.*` / `turn_state.*`. No logic change.

**Step 5: Run to verify GREEN + no behaviour change**

```bash
python -m pytest tests/unit/test_interview_orchestrator_locsize.py -q
python -m pytest tests/integration/test_interview_adaptive_step.py tests/unit/test_interview_decision_invariants.py tests/unit/test_interview_runtime_state.py -q
ruff check abridgeai/features/interviews && pyright abridgeai/features/interviews/orchestrator
```
Expected: LOC guard PASS; all moved-code tests PASS unchanged.

**Step 6: Commit**

```bash
git add abridgeai/features/interviews/orchestrator/
git commit -m "refactor(interviews): split adaptive.py into turn_perception/turn_state (no behaviour change)"
```

---

### Task 7.2: Pure phase-policy module

**Objective:** Encode phase transitions as a pure function of state signals: current phase, turns-in-phase, coverage sufficiency, time remaining, and whether depth budget remains. No DB, no LLM.

**Files:**
- Create: `abridgeai/features/interviews/orchestrator/phases.py`
- Test: `tests/unit/test_interview_phases.py` (new)

**Step 1: Write failing test**

```python
# tests/unit/test_interview_phases.py
from abridgeai.features.interviews.orchestrator.phases import (
    PhaseInputs, next_phase, phase_difficulty_bias,
)
from abridgeai.features.interviews.orchestrator.state import InterviewPhase


def _inp(**kw):
    base = dict(
        current_phase=InterviewPhase.OPENING,
        turns_in_phase=0,
        all_required_covered=False,
        time_fraction_remaining=1.0,
        depth_budget_remaining=True,
        warmup_turns_target=1,
    )
    base.update(kw)
    return PhaseInputs(**base)


def test_opening_advances_to_warmup_after_first_turn():
    assert next_phase(_inp(current_phase=InterviewPhase.OPENING, turns_in_phase=1)) is InterviewPhase.WARMUP


def test_warmup_advances_to_core_after_target():
    assert next_phase(_inp(current_phase=InterviewPhase.WARMUP, turns_in_phase=1)) is InterviewPhase.CORE


def test_core_enters_deep_probe_when_covered_and_time_and_budget():
    got = next_phase(_inp(current_phase=InterviewPhase.CORE, all_required_covered=True,
                          time_fraction_remaining=0.6, depth_budget_remaining=True))
    assert got is InterviewPhase.DEEP_PROBE


def test_core_closes_when_covered_but_low_time():
    got = next_phase(_inp(current_phase=InterviewPhase.CORE, all_required_covered=True,
                          time_fraction_remaining=0.05))
    assert got is InterviewPhase.CLOSING


def test_deep_probe_closes_when_budget_gone_or_low_time():
    assert next_phase(_inp(current_phase=InterviewPhase.DEEP_PROBE, depth_budget_remaining=False)) is InterviewPhase.CLOSING
    assert next_phase(_inp(current_phase=InterviewPhase.DEEP_PROBE, time_fraction_remaining=0.05)) is InterviewPhase.CLOSING


def test_phase_difficulty_bias_monotonic():
    assert phase_difficulty_bias(InterviewPhase.WARMUP) < phase_difficulty_bias(InterviewPhase.CORE)
    assert phase_difficulty_bias(InterviewPhase.CORE) < phase_difficulty_bias(InterviewPhase.DEEP_PROBE)
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_interview_phases.py -q`
Expected: FAIL — module missing.

**Step 3: Implement `phases.py`**

```python
"""Deterministic phase progression (Slice 7).

Pure policy — NO DB, NO LLM. Maps runtime signals to the NEXT interview phase
and a per-phase difficulty bias the selector applies on top of the streak
target. OPENING → WARMUP → CORE → DEEP_PROBE → CLOSING. DEEP_PROBE is entered
only when all required outcomes are provisionally covered AND time + follow-up
budget remain — i.e. the candidate has cleared the bar and we push for ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

from abridgeai.features.interviews.orchestrator.state import InterviewPhase

_LOW_TIME = 0.1        # below this, head to closing regardless of phase
_DEEP_PROBE_MIN_TIME = 0.15  # need at least this much time to open a depth pass


@dataclass(frozen=True)
class PhaseInputs:
    current_phase: InterviewPhase
    turns_in_phase: int
    all_required_covered: bool
    time_fraction_remaining: float | None
    depth_budget_remaining: bool
    warmup_turns_target: int = 1


def _time(x: float | None) -> float:
    return 1.0 if x is None else x


def next_phase(inp: PhaseInputs) -> InterviewPhase:
    t = _time(inp.time_fraction_remaining)
    if inp.current_phase is InterviewPhase.OPENING:
        return InterviewPhase.WARMUP if inp.turns_in_phase >= 1 else InterviewPhase.OPENING
    if inp.current_phase is InterviewPhase.WARMUP:
        if inp.turns_in_phase >= inp.warmup_turns_target:
            return InterviewPhase.CORE
        return InterviewPhase.WARMUP
    if inp.current_phase is InterviewPhase.CORE:
        if inp.all_required_covered:
            if t >= _DEEP_PROBE_MIN_TIME and inp.depth_budget_remaining:
                return InterviewPhase.DEEP_PROBE
            return InterviewPhase.CLOSING
        if t <= _LOW_TIME:
            return InterviewPhase.CLOSING
        return InterviewPhase.CORE
    if inp.current_phase is InterviewPhase.DEEP_PROBE:
        if not inp.depth_budget_remaining or t <= _DEEP_PROBE_MIN_TIME:
            return InterviewPhase.CLOSING
        return InterviewPhase.DEEP_PROBE
    return inp.current_phase  # CLOSING / COMPLETED are terminal here


_DIFFICULTY_BIAS = {
    InterviewPhase.OPENING: -1,
    InterviewPhase.WARMUP: -1,
    InterviewPhase.CORE: 0,
    InterviewPhase.DEEP_PROBE: 1,
    InterviewPhase.CLOSING: -1,
    InterviewPhase.COMPLETED: 0,
}


def phase_difficulty_bias(phase: InterviewPhase) -> int:
    """Signed nudge (−1..+1) added to the streak difficulty target."""
    return _DIFFICULTY_BIAS.get(phase, 0)


__all__ = ["PhaseInputs", "next_phase", "phase_difficulty_bias"]
```

**Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_interview_phases.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add abridgeai/features/interviews/orchestrator/phases.py tests/unit/test_interview_phases.py
git commit -m "feat(interviews): pure phase-progression policy (warmup/deep_probe)"
```

---

### Task 7.3: Track turns-in-phase in runtime state

**Objective:** Add `turns_in_phase: int` and `warmup_turns_target: int` to `InterviewRuntimeStateData`, defaulting safely, and bump `STATE_SCHEMA_VERSION` to 4.

**Files:**
- Modify: `abridgeai/features/interviews/orchestrator/state.py`
- Test: `tests/unit/test_interview_runtime_state.py`

**Step 1: Write failing test**

```python
def test_turns_in_phase_defaults_and_roundtrips():
    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
    d = InterviewRuntimeStateData()
    assert d.turns_in_phase == 0
    assert d.warmup_turns_target == 1
    # tolerant load of an OLD row without the field
    loaded = InterviewRuntimeStateData.from_dict({"phase": "core"})
    assert loaded.turns_in_phase == 0
    # roundtrip
    d.turns_in_phase = 3
    assert InterviewRuntimeStateData.from_dict(d.to_dict()).turns_in_phase == 3
```

**Step 2: Run** → FAIL (attr missing).

**Step 3: Implement** — add fields after `total_follow_up_count`, add to `to_dict`/`from_dict` with `int(data.get("turns_in_phase", 0))` etc., bump `STATE_SCHEMA_VERSION = 4`.

**Step 4: Run** → PASS. Also run `test_interview_runtime_state.py` fully to confirm backfill/tolerance intact.

**Step 5: Commit**

```bash
git add abridgeai/features/interviews/orchestrator/state.py tests/unit/test_interview_runtime_state.py
git commit -m "feat(interviews): persist turns_in_phase + warmup target (schema v4)"
```

---

### Task 7.4: Feed phase into the decision + selection context

**Objective:** In `run_adaptive_turn`, when the v2 `phases` feature is enabled for the session mode, compute `next_phase(...)`, apply `phase_difficulty_bias` to the `student_level` before building `SelectionContext`, and record `turns_in_phase` transitions in `turn_state._apply_state_updates`. When the flag is OFF, behaviour is identical to v1.

**Files:**
- Modify: `abridgeai/features/interviews/orchestrator/adaptive.py`
- Modify: `abridgeai/features/interviews/orchestrator/turn_state.py`
- Test: `tests/integration/test_interview_adaptive_step.py` (new phase-progression case)

**Step 1: Write failing integration test** — drive a session with the `phases` flag on through opening→warmup→core and assert the persisted `data.phase` walks WARMUP then CORE, and that the difficulty target is biased down in warmup. (Mirror the existing adaptive-step harness in that file; pass a settings copy with `adaptive_interviewer_v2_enabled=True, adaptive_v2_phases_enabled=True`.)

**Step 2: Run** → FAIL.

**Step 3: Implement** — thread a `phases_enabled: bool` param into `run_adaptive_turn` (resolved by the caller from settings, Task 7.5), gate the bias + phase transition on it. Clamp biased level to 1..3. Increment/reset `turns_in_phase` in `_apply_state_updates` (reset to 0 on phase change, else +1).

**Step 4: Run** → PASS + full adaptive suite green with flag OFF (regression guard).

**Step 5: Commit**

```bash
git add abridgeai/features/interviews/orchestrator/
git commit -m "feat(interviews): phase-aware difficulty bias + transitions (flag-gated)"
```

---

### Task 7.5: Wire the phases flag through the taking service

**Objective:** Resolve `settings.adaptive_v2_feature_enabled(mode, "phases")` in `services/taking.py` where `run_adaptive_turn` is called, and pass it in.

**Files:**
- Modify: `abridgeai/features/interviews/services/taking.py` (~line 1350 call site + the shadow call ~1445)
- Test: `tests/integration/test_interview_adaptive_step.py`

**Steps:** Standard RED→GREEN. Confirm flag OFF → v1 path identical. Confirm the shadow path also receives the flag so shadowing exercises v2. Commit:

```bash
git commit -am "feat(interviews): resolve v2 phases flag at taking-service call site"
```

**End of Slice 7 — push:**

```bash
python -m pytest tests/unit tests/integration -k "interview or adaptive" -q   # full green vs baseline
git push origin master
```

---

## Slice 8 — Depth probing on strong answers (highest realism ROI)

Today a strong answer just advances. Add probe triggers that dig into excellent answers to find the ceiling — the single biggest realism gap.

### Task 8.1: New probe types

**Objective:** Add `ProbeType.EXTEND_STRONG` and `ProbeType.PROBE_EDGE_CASE` (analysis), and matching `InterviewerActionType.EXTEND_ANSWER` / `PROBE_EDGE_CASE` + `ReasonCode.STRONG_ANSWER_DEPTH_PROBE`.

**Files:**
- Modify: `abridgeai/features/interviews/orchestrator/analysis.py` (ProbeType enum)
- Modify: `abridgeai/features/interviews/orchestrator/decision.py` (action, reason, `_probe_action`/`_probe_reason` maps)
- Test: `tests/unit/test_interview_decision_selection.py`

**Steps:** Add enum members, extend the two probe-mapping dicts. RED test asserts the new ProbeType maps to the new action/reason. GREEN. Commit:

```bash
git commit -am "feat(interviews): add depth probe types (extend_strong, edge_case)"
```

---

### Task 8.2: Decision rule — probe strong answers before advancing

**Objective:** New deterministic rule (numbered 11.5, BEFORE the advance branch): when the answer `is_strong_answer`, the session is in CORE or DEEP_PROBE, follow-up budget remains, and time is not low → return a depth probe instead of advancing. Gated by a `depth_probe_enabled` input on `DecisionInputs` (default False → v1 behaviour).

**Files:**
- Modify: `abridgeai/features/interviews/orchestrator/decision.py`
- Test: `tests/unit/test_interview_decision_invariants.py`

**Step 1: Write failing test**

```python
def test_strong_answer_triggers_depth_probe_when_enabled(strong_analysis):
    from abridgeai.features.interviews.orchestrator.decision import (
        DecisionInputs, InterviewerActionType, decide_next_action)
    from abridgeai.features.interviews.orchestrator.state import InterviewPhase
    inp = DecisionInputs(
        intent=_answer_intent(), analysis=strong_analysis,
        current_question_follow_up_count=0, total_follow_up_count=0,
        time_fraction_remaining=0.7, has_next_question=True,
        all_required_outcomes_covered=False,
        depth_probe_enabled=True, phase=InterviewPhase.DEEP_PROBE,
    )
    d = decide_next_action(inp)
    assert d.action in (InterviewerActionType.EXTEND_ANSWER, InterviewerActionType.PROBE_EDGE_CASE)
    assert d.should_record_academic_evidence is True
    assert d.should_advance_question is False


def test_strong_answer_advances_when_flag_off(strong_analysis):
    # v1 parity: with depth_probe_enabled=False a strong answer still advances
    ...
```

**Step 2: Run** → FAIL.

**Step 3: Implement** — add `depth_probe_enabled: bool = False` and `phase: InterviewPhase = InterviewPhase.CORE` to `DecisionInputs`. Insert the rule after the existing probe block (rule 11), before the advance (rule 12):

```python
    # 11.5 Depth probe (Slice 8): dig into a STRONG answer to find the ceiling.
    if (
        inputs.depth_probe_enabled
        and analysis is not None
        and is_strong_answer(analysis)
        and inputs.phase in (InterviewPhase.CORE, InterviewPhase.DEEP_PROBE)
        and not followups_exhausted
        and not time_low
    ):
        probe = (
            InterviewerActionType.PROBE_EDGE_CASE
            if inputs.phase is InterviewPhase.DEEP_PROBE
            else InterviewerActionType.EXTEND_ANSWER
        )
        return InterviewerDecision(
            action=probe,
            reason_code=ReasonCode.STRONG_ANSWER_DEPTH_PROBE,
            should_record_academic_evidence=True,
            should_advance_question=False,
            acknowledgement_style=AcknowledgementStyle.POSITIVE,
            internal_rationale="Strong answer; probing for depth/ceiling.",
            tags=["depth_probe", probe.value],
        )
```

Import `is_strong_answer` from `coverage` and `InterviewPhase` from `state`. Add the two new actions to `ADVANCE_ACTIONS`? **No** — they keep the same question, so they must NOT be in `ADVANCE_ACTIONS` (they count as follow-ups, like other probes). Verify `_apply_state_updates` treats them as follow-ups (they fall into the else branch → follow-up counter increments), which correctly bounds them by the existing cap.

**Step 4: Run** → PASS, plus full `test_interview_decision_invariants.py` (loop-protection cap still holds because depth probes consume the follow-up budget).

**Step 5: Commit**

```bash
git commit -am "feat(interviews): probe strong answers for depth before advancing (flag-gated)"
```

---

### Task 8.3: Depth-probe utterance templates + i18n

**Objective:** Deterministic fallback phrasings for `EXTEND_ANSWER` / `PROBE_EDGE_CASE` that never leak an answer (generic "can you generalize / what breaks this / edge case" prompts), in EN + VI.

**Files:**
- Modify: `abridgeai/features/interviews/orchestrator/utterance.py` (`build_fallback_utterance` action→template map)
- Modify: frontend `en.json` + `vi.json` (if any client-rendered labels exist for actions; add keys `interview.action.extend_answer`, `interview.action.probe_edge_case`)
- Test: `tests/unit/test_utterance_language_render.py`

**Steps:** RED test asserts EN and VI fallbacks render non-empty, answer-safe text for both actions. GREEN. Verify the answer-leak guard in `utterance_logic._validated_rewrite` still holds (generic probes have no protected content). Commit:

```bash
git commit -am "feat(interviews): depth-probe utterance templates + EN/VI i18n"
```

---

### Task 8.4: Track depth-probe budget for phase policy

**Objective:** `depth_budget_remaining` for `PhaseInputs` (Task 7.2) = `total_follow_up_count < max_total_follow_ups`. Wire it in `run_adaptive_turn`. Confirm DEEP_PROBE exits to CLOSING once the global follow-up budget is spent.

**Files:** Modify `adaptive.py`; test in `test_interview_adaptive_step.py`. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): feed follow-up budget into deep-probe phase exit"
```

**End of Slice 8 — push** (full suite green, flags OFF parity confirmed):

```bash
git push origin master
```

---

## Slice 9 — Cross-turn memory & contradiction

Give the interviewer memory of prior claims per outcome so it can say "earlier you said X — how does that fit?" and detect contradictions across turns, not just within one answer.

### Task 9.1: Bounded claims log on OutcomeCoverageState

**Objective:** Add `claims: list[str]` (bounded, e.g. last 3, ≤200 chars each) to `OutcomeCoverageState`, tolerant deserialization, schema bump to 5.

**Files:** `state.py`; test `test_interview_runtime_state.py`. RED→GREEN. Enforce bound on write in Task 9.3. Commit:

```bash
git commit -am "feat(interviews): bounded per-outcome claims log (schema v5)"
```

---

### Task 9.2: Feed prior claims into answer analysis prompt

**Objective:** Pass the current outcome's prior `claims` into `analyze_answer` so the LLM can flag a cross-turn contradiction (populating `contradictions` + a `RESOLVE_CONTRADICTION` probe). Gated by `cross_turn_enabled`; when off, pass no prior claims (v1 parity).

**Files:**
- Modify: `abridgeai/features/interviews/orchestrator/analysis_logic.py` (prompt input)
- Modify: the analysis prompt template `prompts/*.j2` (add an optional `prior_claims` block)
- Modify: `adaptive.py` (assemble prior claims, gate on flag)
- Test: `tests/unit/test_interview_intent_analysis.py` (analysis parses cross-turn contradiction) + a stub-gateway integration case

**Steps:** RED (analysis with prior-claims context yields a contradiction probe on a stubbed contradictory answer) → GREEN. Keep the prompt answer-safe: prior claims are the STUDENT's own words, not rubric content. Commit:

```bash
git commit -am "feat(interviews): feed prior claims into analysis for cross-turn contradiction (flag-gated)"
```

---

### Task 9.3: Record claims after each analyzed answer

**Objective:** In `turn_state._apply_state_updates`, when analysis has evidence for an outcome, append a short claim summary (from `analysis.identified_concepts` or evidence summary, truncated) to that outcome's `claims`, bounded to the last 3. Gated on `cross_turn_enabled`.

**Files:** `turn_state.py`; test `test_interview_decision_selection.py` or a dedicated state test. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): append bounded per-outcome claims after analysis"
```

**End of Slice 9 — push:**

```bash
git push origin master
```

---

## Slice 10 — Affect / rapport signals (phrasing only, control-flow-safe)

Read lightweight candidate affect (nervous / terse / rambling / confident) and adapt only the utterance TONE — never control flow. This keeps the state machine deterministic while making the interviewer feel human.

### Task 10.1: Affect enum + extend CandidateSignals

**Objective:** Add an `Affect` enum (`NEUTRAL, NERVOUS, TERSE, RAMBLING, CONFIDENT`) and `last_affect: str | None` + booleans on `CandidateSignals`. Schema bump to 6.

**Files:** `intent.py` (or a new `affect.py`), `state.py`; tests. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): affect signal types + candidate-signal fields (schema v6)"
```

---

### Task 10.2: Deterministic affect heuristics + optional LLM read

**Objective:** Pure heuristic first (very short answer → TERSE; very long + low specificity → RAMBLING; hedging phrases → NERVOUS; high confidence + strong → CONFIDENT), LLM read optional and best-effort. Gated by `affect_enabled`.

**Files:** new `abridgeai/features/interviews/orchestrator/affect.py` (pure), `intent_logic.py` (optional LLM), `adaptive.py`. Test `tests/unit/test_interview_affect.py` (new). RED→GREEN→commit:

```bash
git commit -am "feat(interviews): affect heuristics (terse/rambling/nervous/confident)"
```

---

### Task 10.3: Tone-adapt the utterance persona from affect

**Objective:** Pass affect into `generate_utterance`/`build_fallback_utterance` so acknowledgement tone warms for NERVOUS, gently steers for RAMBLING, stays crisp for CONFIDENT. Phrasing only — action/reason/question unchanged. EN + VI tone variants.

**Files:** `utterance.py`, `utterance_logic.py`, prompt template, `en.json` + `vi.json`. Test `test_utterance_language_render.py`. RED→GREEN. Confirm the answer-leak guard + question-verbatim guard still pass. Commit:

```bash
git commit -am "feat(interviews): affect-aware utterance tone (EN/VI, phrasing-only)"
```

**End of Slice 10 — push:**

```bash
git push origin master
```

---

## Slice 11 — Laddered hints & adaptive rephrasing

One flat hint / verbatim repeat today. Ladder assistance and vary rephrasing, loop-protected by the existing follow-up caps.

### Task 11.1: Track hint_level + reframe_count in state

**Objective:** Add `hint_level: int` (0..3) and `reframe_count: int` to state, reset on question advance. Schema bump to 7.

**Files:** `state.py`, `turn_state.py` (reset on advance); tests. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): track hint_level + reframe_count (schema v7)"
```

---

### Task 11.2: Escalate hint level on repeated hint requests

**Objective:** In `decision.py`, when intent is `ASK_FOR_HINT` and `hint_ladder_enabled`, choose an escalating hint action (neutral nudge → structural → worked-approach, NEVER the answer) keyed by `hint_level`. Cap at level 3, then advance. Gated; off → single flat hint (v1).

**Files:** `decision.py` (`ReasonCode` variants optional), `turn_state.py` (increment `hint_level`); tests in `test_interview_decision_invariants.py`. RED→GREEN. Verify hints never enter `should_record_academic_evidence=True`. Commit:

```bash
git commit -am "feat(interviews): laddered hint escalation (answer-safe, flag-gated)"
```

---

### Task 11.3: Vary rephrasing on repeat/reframe

**Objective:** On `REFRAME_QUESTION` / repeated `ASK_TO_REPEAT`, select a different phrasing variant keyed by `reframe_count` rather than re-speaking verbatim. EN + VI variants; answer-safe.

**Files:** `decision.py`, `utterance.py`, `en.json` + `vi.json`, `turn_state.py`. Tests. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): adaptive rephrasing variants on reframe/repeat (EN/VI)"
```

**End of Slice 11 — push:**

```bash
git push origin master
```

---

## Slice 12 — Per-outcome difficulty calibration

Move from one global streak to a per-outcome competence estimate so the interviewer pushes hard on strengths and eases on weak areas — matching how real interviewers calibrate per topic.

### Task 12.1: Per-outcome competence on OutcomeCoverageState

**Objective:** Add `competence_estimate: float` (0..1, EWMA of answer quality for that outcome) to `OutcomeCoverageState`. Schema bump to 8. Pure update helper in `difficulty.py`.

**Files:** `state.py`, `difficulty.py`; tests `tests/unit/test_interview_difficulty.py` (new or existing). RED→GREEN→commit:

```bash
git commit -am "feat(interviews): per-outcome competence estimate (EWMA, schema v8)"
```

---

### Task 12.2: Selection targets difficulty per probed outcome

**Objective:** `SelectionContext` gains an optional `outcome_competence: dict[str, float]`; `_difficulty_fit_score` biases the target difficulty toward the competence of the candidate's *linked outcome* when present, else falls back to the global `student_difficulty_level`. Gated by `per_outcome_difficulty_enabled`.

**Files:** `selection.py`, `adaptive.py`; tests `test_interview_decision_selection.py`. RED→GREEN. Confirm coverage priority still dominates difficulty fit (weights unchanged). Commit:

```bash
git commit -am "feat(interviews): per-outcome difficulty calibration in selection (flag-gated)"
```

---

### Task 12.3: Update competence after each analyzed answer

**Objective:** In `turn_state._apply_state_updates`, fold the answer's quality (strong=1.0, weak=0.0, neutral=leave) into the linked outcome's `competence_estimate` via EWMA. Gated.

**Files:** `turn_state.py`, `difficulty.py`; tests. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): EWMA-update per-outcome competence after analysis"
```

**End of Slice 12 — push:**

```bash
git push origin master
```

---

## Slice 13 — Rich closing

Closing is thin today. Add candidate self-reflection and "any questions for me?" handling — deterministic, brief, answer-safe.

### Task 13.1: Closing sub-steps + benign "questions for me" intent

**Objective:** Add `StudentIntent.ASK_INTERVIEWER_QUESTION` (benign, never scored) and closing sub-actions `INVITE_CANDIDATE_QUESTIONS` / `PROMPT_SELF_REFLECTION` / `CLOSE_INTERVIEW`. Deterministic ordering during CLOSING phase, gated by `rich_closing_enabled`.

**Files:** `intent.py` (+ rules for "do you have… / can I ask" EN + VI), `decision.py` (closing sub-state machine), `state.py` (closing sub-step marker if needed). Tests `test_interview_decision_invariants.py`. RED→GREEN. Ensure "questions for me" is in `NON_ACADEMIC_INTENTS`. Commit:

```bash
git commit -am "feat(interviews): rich closing sub-steps + interviewer-question intent (flag-gated)"
```

---

### Task 13.2: Closing utterance templates + i18n

**Objective:** EN + VI templates for the self-reflection prompt, the invite-questions prompt, and a graceful sign-off. Answer-safe (no rubric/answer content).

**Files:** `utterance.py`, `en.json` + `vi.json`. Test `test_utterance_language_render.py`. RED→GREEN→commit:

```bash
git commit -am "feat(interviews): closing utterance templates (EN/VI)"
```

**End of Slice 13 — push:**

```bash
git push origin master
```

---

## Final verification & rollout wiring

### Task 14.1: Extend the rollback-signal report for v2

**Objective:** The Slice-6 `voice_report.py` aggregator already derives rollback triggers from `voice.decision` / `voice.fallback_activated` events. Confirm the new actions (depth probe, hint ladder, closing sub-steps) surface in the action histogram and do NOT trip `question_loop_detected` (they consume the follow-up budget). Add an action-histogram assertion.

**Files:** `abridgeai/features/interviews/realtime/voice_report.py` (if a hardcoded action set exists), tests. RED→GREEN→commit.

---

### Task 14.2: Update the rollout runbook

**Objective:** Extend `docs/adaptive-interviewer-rollout-runbook.md` with a v2 stage table: v2 is a Stage 6+ that only begins after v1 is at 100% on a mode. Document each sub-flag, the shadow-first order (phases → depth_probe → hint_ladder → affect → cross_turn → per_outcome_difficulty → rich_closing), and that `adaptive_interviewer_v2_enabled=false` is the one-flip v2 kill switch (reverting to v1, which reverts to legacy).

**Files:** `docs/adaptive-interviewer-rollout-runbook.md`. Commit:

```bash
git commit -am "docs(interviews): add adaptive v2 staged-rollout section"
```

---

### Task 14.3: Full-suite green + parity proof

**Objective:** Prove the whole feature is green and that with ALL v2 flags OFF the adaptive path is byte-for-byte v1.

```bash
cd /root/co4029/backend && . .venv/bin/activate
python -m pytest tests/unit tests/integration -k "interview or adaptive" -q
ruff check abridgeai/features/interviews
pyright abridgeai/features/interviews
python -m pytest tests/unit/test_interview_orchestrator_locsize.py -q   # no god files
```

Expected: all pass; LOC guard green; baseline (Task 0.1) has no regressions. Then final push:

```bash
git push origin master
```

---

## Execution notes / pitfalls

- **Never run mypy** on this env (INTERNAL ERROR on Py3.14) — pyright + ruff + pytest only.
- **Every user-facing string needs both `en.json` and `vi.json`.** A missing VI key is a blocker.
- **`adaptive.py` LOC:** Task 7.1 must land first or Slices 8–12 will breach the 800-LOC cap; keep new logic in the pure sibling modules (`phases.py`, `affect.py`, `difficulty.py`), not inlined into `adaptive.py`.
- **Depth probes and hints consume the follow-up budget** — this is what keeps loop-protection invariants (Slice 1) intact. Verify `test_interview_decision_invariants.py` after Slices 8 and 11.
- **Idempotency + one-version-owner:** never add a second `state_repo.save` or a nested commit; extend `_apply_state_updates` in place.
- **Post-session evaluator independence:** none of these provisional signals (competence, claims, affect, phase) may bind the final verdict.
- **Shadow first:** each sub-feature should run in shadow (compute, don't drive) for a window before going live per the runbook.
- **Commit style:** direct to `master`, no branches, no merges (user standing instruction). Push at each slice boundary.
