from abridgeai.features.quizzes.workers.generation import run_quiz_generation_task
from abridgeai.features.quizzes.workers.timing import sweep_overdue_attempts_task

JOBS = [run_quiz_generation_task, sweep_overdue_attempts_task]

__all__ = ["JOBS", "run_quiz_generation_task", "sweep_overdue_attempts_task"]
