from arq import func

from abridgeai.features.interviews.workers.evaluation import (
    EVALUATION_MAX_TRIES,
    evaluate_interview_session_task,
)
from abridgeai.features.interviews.workers.generation import run_interview_generation_task

JOBS = [
    run_interview_generation_task,
    func(evaluate_interview_session_task, max_tries=EVALUATION_MAX_TRIES),
]

__all__ = [
    "EVALUATION_MAX_TRIES",
    "JOBS",
    "evaluate_interview_session_task",
    "run_interview_generation_task",
]
