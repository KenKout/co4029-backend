"""Capability-specific eval runners.

Each runner takes a scenario's inputs, drives the corresponding production
AI pipeline (under `abridgeai.features.<capability>.ai`) for the requested
backend(s), and returns raw outputs + cost breakdown for the judge stage.

T8.2 ships the runner interface + REGISTRY plus deterministic dry-run
behavior. T8.4 will wire the runners up to real LLM calls via the
production AI pipeline modules.
"""

from __future__ import annotations

from eval.runners.base import CapabilityRunner, RunResult
from eval.runners.gap_report import GapReportRunner
from eval.runners.interview import InterviewRunner
from eval.runners.quiz import QuizRunner

REGISTRY: dict[str, type[CapabilityRunner]] = {
    "quiz_generation": QuizRunner,
    "interview_generation": InterviewRunner,
    "gap_report": GapReportRunner,
}

__all__ = [
    "REGISTRY",
    "CapabilityRunner",
    "GapReportRunner",
    "InterviewRunner",
    "QuizRunner",
    "RunResult",
]
