"""Unit tests for quiz generation live-progress projection (0035).

Covers the two pure pieces that don't need a live pipeline:

* :class:`QuizGenerationProgress` — validates a ``progress_json`` payload
  written by the checkpoint helper (stage/index/total + event log).
* :func:`_generation_run_view` — maps a ``GenerationRun`` (with or
  without ``progress_json``) into ``QuizGenerationRunRead``, tolerating
  a missing/partial/malformed checkpoint without raising.

The DB-touching ``record_stage`` round-trip is covered separately in the
integration suite; here we use a tiny stub row so these stay fast + pure.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from abridgeai.features.quizzes.routers.authoring import _generation_run_view
from abridgeai.features.quizzes.schemas import (
    QuizGenerationProgress,
    QuizGenerationRunRead,
)


def _stub_run(**overrides: object) -> SimpleNamespace:
    """Minimal duck-typed ``GenerationRun`` for the view mapper."""
    base: dict[str, object] = {
        "id": uuid.uuid4(),
        "status": "running",
        "started_at": datetime(2026, 7, 22, 12, 0, 0, tzinfo=UTC),
        "created_at": datetime(2026, 7, 22, 11, 59, 0, tzinfo=UTC),
        "finished_at": None,
        "config_json": {},
        "progress_json": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_progress_schema_round_trips_a_checkpoint_payload() -> None:
    payload = {
        "current_stage": "generation",
        "stage_index": 3,
        "total_stages": 6,
        "updated_at": "2026-07-22T12:00:05+00:00",
        "events": [
            {"stage": "retrieval", "at": "2026-07-22T12:00:01+00:00"},
            {
                "stage": "ideation",
                "at": "2026-07-22T12:00:03+00:00",
                "detail": "42 chunks retrieved",
            },
        ],
    }
    progress = QuizGenerationProgress.model_validate(payload)
    assert progress.current_stage == "generation"
    assert progress.stage_index == 3
    assert progress.total_stages == 6
    assert len(progress.events) == 2
    assert progress.events[1].detail == "42 chunks retrieved"
    assert progress.events[0].detail is None


def test_progress_schema_defaults_when_empty() -> None:
    progress = QuizGenerationProgress.model_validate({})
    assert progress.current_stage is None
    assert progress.stage_index == 0
    assert progress.total_stages == 0
    assert progress.events == []


def test_run_view_surfaces_progress_when_present() -> None:
    quiz_id = uuid.uuid4()
    run = _stub_run(
        progress_json={
            "current_stage": "validation",
            "stage_index": 4,
            "total_stages": 6,
            "updated_at": "2026-07-22T12:00:07+00:00",
            "events": [{"stage": "validation", "at": "2026-07-22T12:00:07+00:00"}],
        }
    )
    view = _generation_run_view(run, quiz_id)
    assert isinstance(view, QuizGenerationRunRead)
    assert view.progress is not None
    assert view.progress.current_stage == "validation"
    assert view.progress.stage_index == 4
    assert view.progress.total_stages == 6


def test_run_view_progress_none_when_no_checkpoint() -> None:
    """A run that has never checkpointed maps to ``progress=None``."""
    view = _generation_run_view(_stub_run(progress_json={}), uuid.uuid4())
    assert view.progress is None


def test_run_view_tolerates_malformed_progress() -> None:
    """A malformed checkpoint must not 500 the poll — it degrades to None."""
    run = _stub_run(progress_json={"stage_index": "not-an-int", "events": "not-a-list"})
    view = _generation_run_view(run, uuid.uuid4())
    assert view.progress is None


def test_run_view_missing_progress_attr_is_safe() -> None:
    """A run object without the column (e.g. pre-0035 mock) stays safe."""
    run = _stub_run()
    del run.progress_json
    view = _generation_run_view(run, uuid.uuid4())
    assert view.progress is None
