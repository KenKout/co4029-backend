from arq import func

from abridgeai.features.interviews.workers.analysis import (
    RECONCILE_TURN_ANALYSIS_TASK,
    reconcile_turn_analysis_task,
)
from abridgeai.features.interviews.workers.evaluation import (
    EVALUATION_MAX_TRIES,
    evaluate_interview_session_task,
)
from abridgeai.features.interviews.workers.generation import (
    GENERATION_MAX_TRIES,
    run_interview_generation_task,
)
from abridgeai.features.interviews.workers.recording import (
    reconcile_interview_recordings_task,
)

JOBS = [
    func(run_interview_generation_task, max_tries=GENERATION_MAX_TRIES),
    func(evaluate_interview_session_task, max_tries=EVALUATION_MAX_TRIES),
    reconcile_turn_analysis_task,
    reconcile_interview_recordings_task,
]

__all__ = [
    "EVALUATION_MAX_TRIES",
    "GENERATION_MAX_TRIES",
    "JOBS",
    "RECONCILE_TURN_ANALYSIS_TASK",
    "evaluate_interview_session_task",
    "reconcile_interview_recordings_task",
    "reconcile_turn_analysis_task",
    "run_interview_generation_task",
]
