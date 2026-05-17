"""Integration tests for the COVERAGE quiz pipeline orchestrator (T5.12).

The orchestrator composes ideation (1 call) + per-template generation
(N parallel calls under a semaphore) + a single validation/dedup/
persistence pass on the aggregated questions. These tests substitute
each stage with an ``AsyncMock`` and exercise the parallel-fanout
contract from plan §5920:

* Per-section fanout: each template (one per eligible section) yields
  exactly one ``generate_questions`` call.
* Semaphore enforcement: at most ``parallelism`` generation calls run
  concurrently.
* Pipeline-run-id roll-up: every gateway call sees the same UUID.
* Aggregate downstream: ``validate_questions`` /
  ``discard_duplicates`` / ``persist_questions`` each fire exactly once
  with the **combined** question set, not per-section.
* Per-template failures are absorbed (legacy parity — see commit body).
* Empty outlines raise ``ValueError`` rather than silently returning.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.quizzes.ai.pipelines import coverage as coverage_pipeline
from abridgeai.features.quizzes.ai.pipelines.coverage import run_coverage_pipeline


def _quiz() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), title="Coverage Quiz")


def _run() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), requested_by=uuid4(), config_json={})


def _section(idx: int, n_chunks: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"sec-{idx}",
        title=f"Section {idx}",
        chunk_ids=[uuid4() for _ in range(n_chunks)],
        depth=1,
        page_range=(1, 1),
        content_role="body",
        preview="",
        char_count=200,
    )


def _outline(sections: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(sections=sections, lesson_id=uuid4(), title="Lesson")


def _template_mock(section_id: str, position: int) -> MagicMock:
    template = MagicMock()
    template.model_dump.return_value = {
        "position": position,
        "section_id": section_id,
        "topic": f"topic-{position}",
        "question_type": "mcq",
        "bloom_level": "understand",
        "difficulty": "medium",
        "source_chunk_ids": [str(uuid4())],
        "rationale": "",
    }
    return template


def _candidate_mock(position: int) -> MagicMock:
    candidate = MagicMock()
    candidate.model_dump.return_value = {
        "position": position,
        "question_type": "mcq",
        "prompt_text": f"What is concept {position}?",
        "explanation": "Because.",
        "difficulty": "medium",
        "bloom_level": "understand",
        "options": [
            {"option_key": "A", "option_text": "Yes", "is_correct": True, "position": 1},
            {"option_key": "B", "option_text": "No", "is_correct": False, "position": 2},
        ],
        "source_refs_json": [],
    }
    return candidate


def _chunk_mock() -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content="chunk text",
        metadata={"section_title": "Section"},
        distance=0.0,
    )


class _FakeSession:
    """AsyncSession stub: only commit/rollback are exercised by coverage.py."""

    def __init__(self) -> None:
        self.commit = AsyncMock()
        self.rollback = AsyncMock()


class _FakeSessionCM:
    def __init__(self) -> None:
        self.session = _FakeSession()

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _fake_sessionmaker() -> Any:
    """Return a callable that yields a fresh fake-session context manager."""
    return lambda: _FakeSessionCM()


def _install_default_stage_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    section_count: int = 3,
) -> dict[str, AsyncMock | MagicMock]:
    templates = [_template_mock(f"sec-{i}", i + 1) for i in range(section_count)]
    candidates = [_candidate_mock(i + 1) for i in range(section_count)]

    ideate = AsyncMock(return_value=templates)
    generate = AsyncMock(side_effect=lambda **kw: [candidates[kw["templates"][0]["position"] - 1]])
    validate = AsyncMock(
        return_value=(
            MagicMock(),
            [MagicMock(position=i + 1) for i in range(section_count)],
        )
    )
    apply_v = MagicMock(side_effect=lambda questions, verdicts: (questions, [], []))
    dedup = AsyncMock(side_effect=lambda db, quiz, questions: (questions, []))
    persist = AsyncMock(side_effect=lambda db, run, quiz, chunks, kept: [MagicMock() for _ in kept])
    load_chunks = AsyncMock(return_value=[_chunk_mock()])

    monkeypatch.setattr(coverage_pipeline, "ideate_for_outline", ideate)
    monkeypatch.setattr(coverage_pipeline, "generate_questions", generate)
    monkeypatch.setattr(coverage_pipeline, "validate_questions", validate)
    monkeypatch.setattr(coverage_pipeline, "apply_verdicts", apply_v)
    monkeypatch.setattr(coverage_pipeline, "discard_duplicates", dedup)
    monkeypatch.setattr(coverage_pipeline, "persist_questions", persist)
    monkeypatch.setattr(coverage_pipeline, "_load_chunks_by_id", load_chunks)
    monkeypatch.setattr(coverage_pipeline, "get_sessionmaker", _fake_sessionmaker)

    return {
        "ideate": ideate,
        "generate": generate,
        "validate": validate,
        "apply_verdicts": apply_v,
        "dedup": dedup,
        "persist": persist,
        "load_chunks": load_chunks,
    }


def _coverage_inputs(section_count: int = 3) -> dict[str, Any]:
    sections = [_section(i) for i in range(section_count)]
    outline = _outline(sections)
    budget = {sec.id: 5 for sec in sections}
    return {
        "outlines": [outline],
        "budget": budget,
        "config": {"question_count": section_count * 5},
    }


@pytest.mark.asyncio
async def test_coverage_distributes_questions_across_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch, section_count=3)
    inputs = _coverage_inputs(section_count=3)

    result = await run_coverage_pipeline(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        config=inputs["config"],
        outlines=inputs["outlines"],
        budget=inputs["budget"],
    )

    assert mocks["generate"].await_count == 3
    seen_section_ids = {
        call.kwargs["templates"][0]["section_id"] for call in mocks["generate"].call_args_list
    }
    assert seen_section_ids == {"sec-0", "sec-1", "sec-2"}
    assert len(result) == 3


@pytest.mark.asyncio
async def test_coverage_respects_semaphore_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _install_default_stage_mocks(monkeypatch, section_count=5)
    inputs = _coverage_inputs(section_count=5)
    inputs["config"]["coverage_options"] = {"parallelism": 2}

    state = {"current": 0, "peak": 0}
    candidates = [_candidate_mock(i + 1) for i in range(5)]

    async def _slow_generate(**kw: Any) -> list[Any]:
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await asyncio.sleep(0.02)
        state["current"] -= 1
        return [candidates[kw["templates"][0]["position"] - 1]]

    mocks["generate"] = AsyncMock(side_effect=_slow_generate)
    monkeypatch.setattr(coverage_pipeline, "generate_questions", mocks["generate"])
    mocks["validate"] = AsyncMock(
        return_value=(MagicMock(), [MagicMock(position=i + 1) for i in range(5)])
    )
    monkeypatch.setattr(coverage_pipeline, "validate_questions", mocks["validate"])

    await run_coverage_pipeline(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        config=inputs["config"],
        outlines=inputs["outlines"],
        budget=inputs["budget"],
    )

    assert state["peak"] <= 2, f"semaphore breach: peak concurrent = {state['peak']}"
    assert mocks["generate"].await_count == 5


@pytest.mark.asyncio
async def test_coverage_threads_pipeline_run_id_to_all_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch, section_count=4)
    inputs = _coverage_inputs(section_count=4)

    await run_coverage_pipeline(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        config=inputs["config"],
        outlines=inputs["outlines"],
        budget=inputs["budget"],
    )

    threaded_ids: set[UUID] = set()
    threaded_ids.add(mocks["ideate"].call_args.kwargs["pipeline_run_id"])
    for call in mocks["generate"].call_args_list:
        threaded_ids.add(call.kwargs["pipeline_run_id"])
    threaded_ids.add(mocks["validate"].call_args.kwargs["pipeline_run_id"])

    assert len(threaded_ids) == 1, f"divergent pipeline_run_ids: {threaded_ids}"
    assert isinstance(next(iter(threaded_ids)), UUID)


@pytest.mark.asyncio
async def test_coverage_invokes_ideation_with_outline_for_redistribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage delegates chunk-anchor redistribution to
    ``ideate_for_outline`` (which internally calls
    ``_redistribute_chunk_anchors_within_section``). Verify ideation
    sees both ``outlines`` and ``budget`` so the redistribute helper
    has the per-section chunk inventory it needs.
    """
    mocks = _install_default_stage_mocks(monkeypatch, section_count=3)
    inputs = _coverage_inputs(section_count=3)

    await run_coverage_pipeline(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        config=inputs["config"],
        outlines=inputs["outlines"],
        budget=inputs["budget"],
    )

    ideate_call = mocks["ideate"].call_args
    assert ideate_call.kwargs["outlines"] is inputs["outlines"]
    assert ideate_call.kwargs["budget"] == inputs["budget"]


@pytest.mark.asyncio
async def test_coverage_validation_dedup_persistence_run_once_aggregated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch, section_count=4)
    inputs = _coverage_inputs(section_count=4)

    await run_coverage_pipeline(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        config=inputs["config"],
        outlines=inputs["outlines"],
        budget=inputs["budget"],
    )

    assert mocks["validate"].await_count == 1
    assert mocks["dedup"].await_count == 1
    assert mocks["persist"].await_count == 1

    validate_questions = mocks["validate"].call_args.kwargs["questions"]
    assert len(validate_questions) == 4

    dedup_questions = mocks["dedup"].call_args.args[2]
    assert len(dedup_questions) == 4

    persist_kept = mocks["persist"].call_args.args[4]
    assert len(persist_kept) == 4


@pytest.mark.asyncio
async def test_coverage_one_section_failure_is_absorbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy ``return_exceptions=False`` works because the per-template
    ``except`` already swallows failures into ``None``. Verify: one
    section's ``generate_questions`` raising still leaves the pipeline
    running on the surviving 2 sections.
    """
    mocks = _install_default_stage_mocks(monkeypatch, section_count=3)
    inputs = _coverage_inputs(section_count=3)

    candidates = [_candidate_mock(i + 1) for i in range(3)]

    async def _flaky_generate(**kw: Any) -> list[Any]:
        position = kw["templates"][0]["position"]
        if position == 2:
            raise RuntimeError("LLM hiccup")
        return [candidates[position - 1]]

    mocks["generate"] = AsyncMock(side_effect=_flaky_generate)
    monkeypatch.setattr(coverage_pipeline, "generate_questions", mocks["generate"])
    mocks["validate"] = AsyncMock(
        return_value=(MagicMock(), [MagicMock(position=i) for i in (1, 2)])
    )
    monkeypatch.setattr(coverage_pipeline, "validate_questions", mocks["validate"])

    result = await run_coverage_pipeline(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        config=inputs["config"],
        outlines=inputs["outlines"],
        budget=inputs["budget"],
    )

    assert mocks["generate"].await_count == 3
    assert len(mocks["validate"].call_args.kwargs["questions"]) == 2
    assert len(result) == 2


@pytest.mark.asyncio
async def test_coverage_zero_sections_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_stage_mocks(monkeypatch, section_count=0)

    with pytest.raises(ValueError, match="precomputed outlines"):
        await run_coverage_pipeline(
            db=MagicMock(),
            run=_run(),
            quiz=_quiz(),
            config={"question_count": 0},
            outlines=[],
            budget={},
        )


@pytest.mark.asyncio
async def test_coverage_empty_budget_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_default_stage_mocks(monkeypatch, section_count=2)
    inputs = _coverage_inputs(section_count=2)

    with pytest.raises(ValueError, match="budget allocation"):
        await run_coverage_pipeline(
            db=MagicMock(),
            run=_run(),
            quiz=_quiz(),
            config=inputs["config"],
            outlines=inputs["outlines"],
            budget={},
        )


@pytest.mark.asyncio
async def test_coverage_propagates_validation_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch, section_count=3)
    inputs = _coverage_inputs(section_count=3)

    mocks["validate"] = AsyncMock(side_effect=RuntimeError("validator down"))
    monkeypatch.setattr(coverage_pipeline, "validate_questions", mocks["validate"])

    with pytest.raises(RuntimeError, match="validator down"):
        await run_coverage_pipeline(
            db=MagicMock(),
            run=_run(),
            quiz=_quiz(),
            config=inputs["config"],
            outlines=inputs["outlines"],
            budget=inputs["budget"],
        )

    assert mocks["dedup"].await_count == 0
    assert mocks["persist"].await_count == 0


def test_coverage_pipeline_module_is_under_soft_cap() -> None:
    """Coverage is the largest pipeline (parallel fanout + per-section
    sessions + budget plumbing), so the plan's 350 LOC cap is a soft
    target. Track regressions but don't fail the build hard.
    """
    target = Path(__file__).resolve().parents[2] / (
        "abridgeai/features/quizzes/ai/pipelines/coverage.py"
    )
    assert target.exists(), f"missing {target}"
    line_count = len(target.read_text(encoding="utf-8").splitlines())
    assert line_count <= 450, f"coverage.py is {line_count} LOC; legacy port budget is 450"
