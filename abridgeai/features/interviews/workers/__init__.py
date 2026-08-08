from arq import func

from abridgeai.features.interviews.workers.analysis import (
    RECONCILE_TURN_ANALYSIS_TASK,
    reconcile_turn_analysis_task,
)
from abridgeai.features.interviews.workers.evaluation import (
    EVALUATION_MAX_TRIES,
    evaluate_interview_session_task,
    generate_practice_feedback_task,
)
from abridgeai.features.interviews.workers.generation import run_interview_generation_task

JOBS = [
    run_interview_generation_task,
    func(evaluate_interview_session_task, max_tries=EVALUATION_MAX_TRIES),
    generate_practice_feedback_task,
    reconcile_turn_analysis_task,
]

__all__ = [
    "EVALUATION_MAX_TRIES",
    "JOBS",
    "RECONCILE_TURN_ANALYSIS_TASK",
    "evaluate_interview_session_task",
    "generate_practice_feedback_task",
    "reconcile_turn_analysis_task",
    "run_interview_generation_task",
]
