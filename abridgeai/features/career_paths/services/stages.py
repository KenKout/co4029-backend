"""Stage evaluation: satisfied → stage complete → unlock → path progress.

This is the single place the staged-pathway rules live. Everything else
(routers, readiness snapshots, the Start endpoint) calls in here rather than
re-deriving any of it.

The rules
---------
``satisfied(course)`` ⟺ ``course_enrollments.status = 'completed'`` (D2),
maintained synchronously by ``enrollments.services.completion``. Note this is
NOT ``completion_percent >= 100``: a course the student never enrolled in has
no status row at all, and lesson progress alone must not satisfy a course.

``stage_complete`` ⟺ latched in ``student_stage_progress``
**OR** (all required satisfied AND satisfied_optional >=
``min_optional_to_complete``). Evaluating true writes the latch, and the
latch wins forever after — see :class:`~..models.StudentStageProgress`.

``unlocked``:

* stage 1 → unconditionally ``True``. Its stored ``unlock_policy`` is inert;
  a path whose first stage is locked could never be started by anyone. This
  is a documented implicit override, deliberately NOT normalised in the
  database, so that reordering a stage away from position 1 restores the
  manager's original intent.
* ``always`` → ``True``
* ``after_previous`` → previous stage complete
* ``after_previous_required`` → previous stage's required courses all
  satisfied (electives may still be outstanding)

Progress::

    stage_total = required_count + min_optional_to_complete
    stage_done  = satisfied_required + min(satisfied_optional, min_optional)
    path_progress = Σ stage_done / max(1, Σ stage_total where stage_total > 0)

Stages with ``stage_total == 0`` are complete and excluded from the
denominator — a stage of pure electives with no quota cannot be "half done",
and including it would let an empty stage drag a finished path below 100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.features.career_paths.queries import authoring as authoring_queries
from abridgeai.features.career_paths.queries import student as student_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.career_paths.models import CareerPathStage

_LEGACY_FORMULA = 1
_STAGE_AWARE_FORMULA = 2

PROGRESS_FORMULA_SETTING = "careerpath.progress_formula_version"


@dataclass
class StageEval:
    """One stage's evaluated state for one student."""

    stage: CareerPathStage
    courses: list[dict[str, Any]] = field(default_factory=list)
    latched: bool = False
    unlocked: bool = False

    @property
    def required(self) -> list[dict[str, Any]]:
        return [c for c in self.courses if c["is_required"]]

    @property
    def optional(self) -> list[dict[str, Any]]:
        return [c for c in self.courses if not c["is_required"]]

    @property
    def satisfied_required(self) -> int:
        return sum(1 for c in self.required if c["satisfied"])

    @property
    def satisfied_optional(self) -> int:
        return sum(1 for c in self.optional if c["satisfied"])

    @property
    def all_required_satisfied(self) -> bool:
        return all(c["satisfied"] for c in self.required)

    @property
    def stage_total(self) -> int:
        return len(self.required) + self.stage.min_optional_to_complete

    @property
    def stage_done(self) -> int:
        return self.satisfied_required + min(
            self.satisfied_optional, self.stage.min_optional_to_complete
        )

    @property
    def live_complete(self) -> bool:
        """The rule, ignoring the latch."""
        return (
            self.all_required_satisfied
            and self.satisfied_optional >= self.stage.min_optional_to_complete
        )

    @property
    def complete(self) -> bool:
        """Latched OR live. Latched wins: completion never goes backward."""
        return self.latched or self.live_complete


async def evaluate_stages(
    db: AsyncSession,
    *,
    version_id: UUID,
    student_id: UUID,
    enrollment_id: UUID | None,
) -> list[StageEval]:
    """Evaluate every stage of ONE VERSION for one student, in position order.

    ``version_id`` is the version the student's enrollment is pinned to
    (Gap 3: their route never changes under them). ``enrollment_id`` may be
    ``None`` for a student who is not enrolled (a manager previewing, or a
    published-path browse). Nothing is latched in that case — there is no
    enrollment to latch against — but unlock and completion still evaluate
    so the preview matches what a student would see.
    """
    stages = await authoring_queries.list_stages_for_version(db, version_id)
    rows = await student_queries.get_path_course_progress(
        db, version_id=version_id, student_id=student_id
    )
    latched = (
        await student_queries.list_latched_stage_ids(db, enrollment_id)
        if enrollment_id is not None
        else set()
    )

    by_stage: dict[UUID, list[dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(row["stage_id"], []).append(row)

    evals = [
        StageEval(
            stage=stage,
            courses=by_stage.get(stage.id, []),
            latched=stage.id in latched,
        )
        for stage in stages
    ]

    _apply_unlock(evals)
    return evals


def _apply_unlock(evals: list[StageEval]) -> None:
    """Set ``unlocked`` on each stage in order.

    Sequential because ``after_previous`` reads the previous stage's
    evaluated completion, which itself may be latched.
    """
    for idx, ev in enumerate(evals):
        if idx == 0:
            # Position 1 is an implicit, documented override: always
            # unlocked, whatever the stored policy says.
            ev.unlocked = True
            continue
        previous = evals[idx - 1]
        policy = ev.stage.unlock_policy
        if policy == "always":
            ev.unlocked = True
        elif policy == "after_previous":
            ev.unlocked = previous.complete
        elif policy == "after_previous_required":
            ev.unlocked = previous.all_required_satisfied
        else:  # unknown policy: fail CLOSED rather than silently opening
            ev.unlocked = False


async def latch_completed_stages(
    db: AsyncSession, *, enrollment_id: UUID, evals: list[StageEval]
) -> int:
    """Write latch rows for stages that evaluate complete but aren't latched.

    Returns how many rows were written, so the caller knows whether to
    commit. Idempotent (``ON CONFLICT DO NOTHING``).
    """
    written = 0
    for ev in evals:
        if ev.latched or not ev.live_complete:
            continue
        if await student_queries.latch_stage_complete(
            db, enrollment_id=enrollment_id, stage_id=ev.stage.id
        ):
            ev.latched = True
            written += 1
    return written


def path_progress_percent(evals: list[StageEval]) -> float:
    """Stage-aware path completion, 0–100 (formula version 2).

    Stages whose ``stage_total`` is 0 are complete by definition and are
    excluded from the denominator entirely.
    """
    counted = [ev for ev in evals if ev.stage_total > 0]
    if not counted:
        # Every stage is empty or quota-free: nothing to measure. A path with
        # no measurable work is complete, not 0% — otherwise a student can
        # never finish it.
        return 100.0
    total = sum(ev.stage_total for ev in counted)
    done = sum(min(ev.stage_done, ev.stage_total) for ev in counted)
    return round(done / max(1, total) * 100, 2)


def legacy_progress_percent(courses: list[dict[str, Any]]) -> float:
    """Formula version 1: flat mean of every course's completion percent.

    Preserved verbatim so the pre-stage behaviour is bit-for-bit reproducible
    while the cutover setting still reads 1.
    """
    if not courses:
        return 0.0
    return sum(float(c["completion_percent"]) for c in courses) / len(courses)


async def resolve_formula_version(db: AsyncSession) -> int:
    """The active progress formula version (global runtime setting).

    Global on purpose: a single dated cutover keeps the readiness chart
    segmentable on time rather than per path. Degrades to the legacy formula
    if the setting cannot be read — never fails a student's progress read
    over a tuning knob.
    """
    from abridgeai.core.runtime_settings import resolve_setting  # noqa: PLC0415

    try:
        return int(await resolve_setting(db, PROGRESS_FORMULA_SETTING))
    except Exception:  # noqa: BLE001 -- tuning knob; fall back to legacy
        return _LEGACY_FORMULA


def stage_is_hard_locked(ev: StageEval) -> bool:
    """Whether a locked stage should actually BLOCK access.

    Only ``enforcement='hard'`` blocks; ``soft`` and ``advisory`` render a
    warning and let the student through. This governs stage lock only — the
    ``max_concurrent`` attention cap never blocks under any enforcement.
    """
    return not ev.unlocked and ev.stage.enforcement == "hard"


__all__ = [
    "PROGRESS_FORMULA_SETTING",
    "StageEval",
    "evaluate_stages",
    "latch_completed_stages",
    "legacy_progress_percent",
    "path_progress_percent",
    "resolve_formula_version",
    "stage_is_hard_locked",
]
