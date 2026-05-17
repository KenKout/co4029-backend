"""Integration tests for the per-question REGENERATE pipeline (T5.11).

The orchestrator composes retrieval (optional) → generation →
validation → dedup → persistence (replace-in-place). Tests substitute
each stage with an ``AsyncMock`` and assert:

* ``replace_question_in_place`` is called with the same question
  instance — the input ``question_id`` round-trips and revision_no is
  bumped (delegated to T5.9, asserted via mock).
* The pipeline-generated ``pipeline_run_id`` is threaded through every
  stage that accepts it (audit-rollup contract from plan §5841).
* Ideation is **never** invoked — regeneration uses the existing
  question's prompt as its template (plan §5889).
* Sibling prompts are forwarded to the generation stage as
  ``previous_questions`` so the LLM avoids reproducing a sibling.
* The orchestrator stays under the 200 LOC ceiling from plan §5885.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.quizzes.ai.pipelines import regenerate as regen_pipeline
from abridgeai.features.quizzes.ai.pipelines.regenerate import run_question_regeneration
from abridgeai.features.quizzes.ai.stages import ideation as ideation_stage


def _quiz() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), title="Photosynthesis Basics")


def _run() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), requested_by=uuid4(), config_json={})


def _question(*, prompt: str = "Original prompt?") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        position=3,
        prompt_text=prompt,
        question_type="mcq",
        bloom_level="apply",
        difficulty="hard",
        source_refs=[{"chunk_id": "c-old", "material_version_id": "m-1"}],
    )


def _kg_context() -> SimpleNamespace:
    return SimpleNamespace(is_empty=True, concepts=[], prerequisites=[], related=[])


def _chunk(idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=f"chunk text {idx}",
        metadata={"section_title": f"Section {idx}"},
    )


def _candidate_mock(position: int = 1) -> MagicMock:
    candidate = MagicMock()
    candidate.model_dump.return_value = {
        "position": position,
        "question_type": "mcq",
        "prompt_text": "Regenerated prompt?",
        "explanation": "Because.",
        "difficulty": "medium",
        "bloom_level": "understand",
        "options": [
            {"option_key": "A", "option_text": "Yes", "is_correct": True, "position": 1},
            {"option_key": "B", "option_text": "No", "is_correct": False, "position": 2},
            {"option_key": "C", "option_text": "Maybe", "is_correct": False, "position": 3},
            {"option_key": "D", "option_text": "Never", "is_correct": False, "position": 4},
        ],
        "source_refs_json": ["c-1"],
        "original_generated_payload": {},
    }
    return candidate


def _install_default_stage_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sibling_prompts: list[str] | None = None,
) -> dict[str, AsyncMock | MagicMock]:
    chunks_back = [_chunk(1), _chunk(2)]
    retrieve = AsyncMock(return_value=(chunks_back, [0.1, 0.2], ["anchor"]))
    generate = AsyncMock(return_value=[_candidate_mock(1)])
    validate = AsyncMock(return_value=(MagicMock(), [MagicMock(position=1)]))
    apply_v = MagicMock(side_effect=lambda questions, verdicts: (questions, [], []))
    dedup = AsyncMock(side_effect=lambda db, quiz, questions: (questions, []))
    replace = AsyncMock(side_effect=lambda db, run, q, payload, chunks=None: q)
    fetch_siblings = AsyncMock(return_value=list(sibling_prompts or []))
    ideate = AsyncMock(return_value=[])

    monkeypatch.setattr(regen_pipeline, "retrieve_chunks", retrieve)
    monkeypatch.setattr(regen_pipeline, "generate_questions", generate)
    monkeypatch.setattr(regen_pipeline, "validate_questions", validate)
    monkeypatch.setattr(regen_pipeline, "apply_verdicts", apply_v)
    monkeypatch.setattr(regen_pipeline, "discard_duplicates", dedup)
    monkeypatch.setattr(regen_pipeline, "replace_question_in_place", replace)
    monkeypatch.setattr(regen_pipeline, "_fetch_sibling_prompts", fetch_siblings)
    monkeypatch.setattr(ideation_stage, "ideate_for_outline", ideate)

    return {
        "retrieve": retrieve,
        "generate": generate,
        "validate": validate,
        "apply_verdicts": apply_v,
        "dedup": dedup,
        "replace": replace,
        "fetch_siblings": fetch_siblings,
        "ideate": ideate,
    }


@pytest.mark.asyncio
async def test_regenerate_replaces_in_place_same_question_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch)
    question = _question()

    result = await run_question_regeneration(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        question=question,
        chunks=[_chunk(1)],
        kg_context=_kg_context(),
        config={"difficulty": "hard"},
    )

    assert result is question, "regenerate must return the same instance"
    replace_call = mocks["replace"].call_args
    passed_question = (
        replace_call.args[2] if len(replace_call.args) >= 3 else replace_call.kwargs["question"]
    )
    assert passed_question is question
    assert passed_question.id == question.id


@pytest.mark.asyncio
async def test_regenerate_threads_pipeline_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch)

    await run_question_regeneration(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        question=_question(),
        chunks=[_chunk(1)],
        kg_context=_kg_context(),
        config={},
    )

    threaded: set[UUID] = set()
    for stage_key in ("generate", "validate"):
        kwargs = mocks[stage_key].call_args.kwargs
        assert "pipeline_run_id" in kwargs, f"{stage_key} did not receive pipeline_run_id"
        threaded.add(kwargs["pipeline_run_id"])

    assert len(threaded) == 1, f"Stages received divergent pipeline_run_ids: {threaded}"
    assert isinstance(next(iter(threaded)), UUID)


@pytest.mark.asyncio
async def test_regenerate_skips_ideation(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _install_default_stage_mocks(monkeypatch)

    await run_question_regeneration(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        question=_question(),
        chunks=[_chunk(1)],
        kg_context=_kg_context(),
        config={},
    )

    assert mocks["ideate"].await_count == 0, "regenerate must NOT invoke ideation (plan §5889)"


@pytest.mark.asyncio
async def test_regenerate_passes_previous_questions_to_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    siblings = ["What is photosynthesis?", "Define chlorophyll."]
    mocks = _install_default_stage_mocks(monkeypatch, sibling_prompts=siblings)

    await run_question_regeneration(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        question=_question(),
        chunks=[_chunk(1)],
        kg_context=_kg_context(),
        config={},
    )

    gen_kwargs = mocks["generate"].call_args.kwargs
    assert gen_kwargs.get("previous_questions") == siblings


@pytest.mark.asyncio
async def test_regenerate_retries_retrieval_when_chunks_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch)
    question = _question(prompt="What is mitosis?")

    await run_question_regeneration(
        db=MagicMock(),
        run=_run(),
        quiz=_quiz(),
        question=question,
        chunks=[],
        kg_context=_kg_context(),
        config={},
    )

    assert mocks["retrieve"].await_count == 1
    retrieve_kwargs = mocks["retrieve"].call_args.kwargs
    assert retrieve_kwargs["question_anchor"] == question.prompt_text


@pytest.mark.asyncio
async def test_regenerate_raises_when_validator_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _install_default_stage_mocks(monkeypatch)
    rejection: dict[str, Any] = {
        "position": 1,
        "question_id": 1,
        "prompt_text": "Bad?",
        "reasons": ["UNGROUNDED"],
        "evidence_excerpt": None,
    }
    mocks["apply_verdicts"] = MagicMock(
        side_effect=lambda questions, verdicts: ([], [rejection], ["UNGROUNDED"])
    )
    monkeypatch.setattr(regen_pipeline, "apply_verdicts", mocks["apply_verdicts"])

    with pytest.raises(ValueError, match="UNGROUNDED"):
        await run_question_regeneration(
            db=MagicMock(),
            run=_run(),
            quiz=_quiz(),
            question=_question(),
            chunks=[_chunk(1)],
            kg_context=_kg_context(),
            config={},
        )

    assert mocks["dedup"].await_count == 0
    assert mocks["replace"].await_count == 0


def test_no_god_file_in_regenerate() -> None:
    target = Path(__file__).resolve().parents[2] / (
        "abridgeai/features/quizzes/ai/pipelines/regenerate.py"
    )
    assert target.exists(), f"missing {target}"
    line_count = len(target.read_text(encoding="utf-8").splitlines())
    assert line_count <= 200, f"regenerate.py is {line_count} LOC; plan §5885 caps at 200"
