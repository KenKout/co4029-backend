"""Unit tests for the interview GENERATION stage (T6.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.ai.llm import LLMResult, LLMRole
from abridgeai.features.interviews.ai.stages.generation import (
    InterviewQuestionDraft,
    generate_interview_questions,
    parse_generation_response,
)


@dataclass
class _FakeChunk:
    chunk_id: UUID = field(default_factory=uuid4)
    content: str = "Recursion is when a function calls itself."


@dataclass
class _FakeContext:
    chunks: list[_FakeChunk] = field(default_factory=lambda: [_FakeChunk()])


def _llm_result(content_json: dict[str, object]) -> LLMResult:
    return LLMResult(
        role=LLMRole.INTERVIEW_GENERATION,
        tier="large",
        model_name="test-model",
        base_url="https://example.test/v1",
        stage_name="interview_generation",
        pipeline_run_id=None,
        request_payload={},
        response_payload={"choices": [{"message": {"content": "{}"}}]},
        content_json=content_json,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cached_input_tokens=None,
        latency_ms=42,
        estimated_cost_usd=Decimal("0.0001"),
    )


def _fake_run(question_count: int | None = None) -> SimpleNamespace:
    config_json: dict[str, object] = {}
    if question_count is not None:
        config_json["question_count"] = question_count
    return SimpleNamespace(id=uuid4(), config_json=config_json)


def _fake_config(supplementary: str | None = None, persona: str = "neutral") -> SimpleNamespace:
    return SimpleNamespace(
        title="Algorithms 101 Interview",
        persona=persona,
        supplementary_instructions=supplementary,
    )


def _fake_outcomes(n: int = 3) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=uuid4(),
            outcome_text=f"Outcome {idx}",
            outcome_type="knowledge",
            importance_weight=3,
        )
        for idx in range(n)
    ]


_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _question(
    *,
    position: int,
    question_type: str,
    difficulty: str,
    expected_depth: int,
    linked_outcome_id: UUID | None = None,
    source_refs: list[str] | None = None,
    logical_question_index: int | None = None,
) -> dict[str, Any]:
    return {
        "position": position,
        "question_type": question_type,
        "prompt_text": f"Q{position}: explain {question_type} concept #{position}",
        "difficulty": difficulty,
        "expected_depth": expected_depth,
        "linked_outcome_id": str(linked_outcome_id) if linked_outcome_id else None,
        "logical_question_index": logical_question_index,
        "source_refs": source_refs or [str(uuid4())],
        "rationale": f"probes #{position}",
    }


def _eight_questions() -> dict[str, Any]:
    return {
        "questions": [
            _question(position=1, question_type="technical", difficulty="easy", expected_depth=1),
            _question(position=2, question_type="technical", difficulty="easy", expected_depth=2),
            _question(position=3, question_type="technical", difficulty="easy", expected_depth=2),
            _question(position=4, question_type="technical", difficulty="medium", expected_depth=3),
            _question(
                position=5, question_type="behavioral", difficulty="medium", expected_depth=3
            ),
            _question(
                position=6, question_type="behavioral", difficulty="medium", expected_depth=3
            ),
            _question(position=7, question_type="situational", difficulty="hard", expected_depth=4),
            _question(position=8, question_type="technical", difficulty="hard", expected_depth=5),
        ]
    }


@pytest.mark.asyncio
async def test_generates_n_questions() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    assert len(drafts) == 8
    assert all(isinstance(d, InterviewQuestionDraft) for d in drafts)
    gateway.generate_json.assert_awaited_once()
    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["role"] is LLMRole.INTERVIEW_GENERATION
    assert kwargs["stage_name"] == "interview_generation"


@pytest.mark.asyncio
async def test_form_question_count_drives_prompt_and_caps_drafts() -> None:
    """The teacher's form count (run.config_json) must be honoured exactly:
    it both reaches the prompt and caps how many drafts are returned."""
    gateway = AsyncMock()
    # LLM returns 8 questions, but the teacher asked for 3.
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(question_count=3),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Total questions to produce: 3" in user_prompt
    assert len(drafts) == 3  # capped to the requested count, not the LLM's 8


@pytest.mark.asyncio
async def test_form_count_overrides_supplementary_default() -> None:
    """When both are present, the form count wins over the legacy
    supplementary-instructions override."""
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(question_count=5),
        config=_fake_config(supplementary='{"question_count": 12}'),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Total questions to produce: 5" in user_prompt


@pytest.mark.asyncio
async def test_type_mix_matches_default_60_30_10() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "technical: 60%" in user_prompt
    assert "behavioral: 30%" in user_prompt
    assert "situational: 10%" in user_prompt


@pytest.mark.asyncio
async def test_type_mix_overridden_by_rubric_weights() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(
            supplementary='{"rubric_weights": {"technical": 70, "behavioral": 20, "situational": 10}}'
        ),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "technical: 70%" in user_prompt
    assert "behavioral: 20%" in user_prompt
    assert "situational: 10%" in user_prompt


@pytest.mark.asyncio
async def test_difficulty_progression() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    levels = [_DIFFICULTY_ORDER[d.difficulty] for d in drafts]
    assert levels == sorted(levels), (
        f"difficulties not non-decreasing: {[d.difficulty for d in drafts]}"
    )
    assert levels[0] <= levels[-1]


@pytest.mark.asyncio
async def test_expected_depth_in_range_1_to_5() -> None:
    payload = {
        "questions": [
            _question(position=1, question_type="technical", difficulty="easy", expected_depth=0),
            _question(position=2, question_type="technical", difficulty="easy", expected_depth=99),
            _question(position=3, question_type="technical", difficulty="easy", expected_depth=3),
        ]
    }
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(payload))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    assert all(1 <= d.expected_depth <= 5 for d in drafts)
    assert drafts[0].expected_depth == 1
    assert drafts[1].expected_depth == 5


@pytest.mark.asyncio
async def test_each_question_links_to_outcome_id() -> None:
    outcomes = _fake_outcomes(3)
    payload = _eight_questions()
    payload["questions"][0]["linked_outcome_id"] = None  # type: ignore[index]
    payload["questions"][3]["linked_outcome_id"] = None  # type: ignore[index]

    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(payload))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=outcomes,
        gateway=gateway,
    )

    linked = sum(1 for d in drafts if d.linked_outcome_id is not None)
    assert linked >= len(drafts) - 1
    valid_ids = {o.id for o in outcomes}
    for draft in drafts:
        if draft.linked_outcome_id is not None:
            assert draft.linked_outcome_id in valid_ids or isinstance(draft.linked_outcome_id, UUID)


def test_parser_drops_malformed_entries() -> None:
    payload: dict[str, Any] = {
        "questions": [
            {"prompt_text": "", "question_type": "technical"},
            "not even a dict",
            {"prompt_text": "Bad type", "question_type": "alien"},
            _question(position=1, question_type="technical", difficulty="easy", expected_depth=2),
        ]
    }

    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    assert parsed[0].question_type == "technical"


def test_parser_drops_non_string_question_type() -> None:
    """A number / list / object in ``question_type`` must not crash ``.strip()``.

    The LLM occasionally emits ``"question_type": 3`` (an index) or a list of
    angles. Before the guard this raised ``AttributeError`` inside the parser
    and failed the whole generation run instead of dropping one row.
    """
    good = _question(position=1, question_type="technical", difficulty="easy", expected_depth=2)
    for bad_value in (3, ["technical"], {"kind": "technical"}, True):
        payload: dict[str, Any] = {
            "questions": [
                {**good, "prompt_text": "Bad type row", "question_type": bad_value},
                good,
            ]
        }
        parsed = parse_generation_response(payload)
        assert [d.prompt_text for d in parsed] == [good["prompt_text"]]


def test_parser_drops_non_string_difficulty() -> None:
    """Same guard for ``difficulty`` — a numeric level is dropped, not fatal."""
    good = _question(position=1, question_type="technical", difficulty="easy", expected_depth=2)
    for bad_value in (2, ["hard"], {"level": "hard"}):
        payload: dict[str, Any] = {
            "questions": [
                {**good, "prompt_text": "Bad difficulty row", "difficulty": bad_value},
                good,
            ]
        }
        parsed = parse_generation_response(payload)
        assert [d.prompt_text for d in parsed] == [good["prompt_text"]]


def test_parser_treats_null_type_and_difficulty_as_defaults() -> None:
    """Explicit ``null`` keeps the historical default, unlike a wrong type."""
    entry = _question(position=1, question_type="technical", difficulty="easy", expected_depth=2)
    entry["question_type"] = None
    entry["difficulty"] = None
    parsed = parse_generation_response({"questions": [entry]})
    assert len(parsed) == 1
    assert parsed[0].question_type == "technical"
    assert parsed[0].difficulty == "medium"


def test_parser_accepts_british_spelling() -> None:
    payload: dict[str, Any] = {
        "questions": [
            {
                "prompt_text": "Tell me about a time you failed.",
                "question_type": "behavioural",
                "difficulty": "medium",
                "expected_depth": 3,
                "linked_outcome_id": None,
                "source_refs": [],
                "rationale": "self-reflection",
            }
        ]
    }
    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    assert parsed[0].question_type == "behavioral"


def test_jinja_prompts_in_j2_files_only() -> None:
    here = Path(__file__).resolve().parents[2]
    target = (
        here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "generation" / "prompts"
    )
    assert target.is_dir(), f"prompts dir missing at {target}"
    j2_files = list(target.glob("*.j2"))
    assert {p.name for p in j2_files} >= {"system.j2", "user.j2"}
    for path in j2_files:
        assert path.stat().st_size > 0, f"empty template: {path}"


def test_no_inline_prompts_in_python() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "generation"
    forbidden_phrases = (
        "You are an interviewer",
        "Return JSON in this shape",
        "Difficulty progression rubric",
    )
    for path in target.glob("*.py"):
        body = path.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            assert phrase not in body, f"{path.name} inlines prompt phrase: {phrase!r}"


def test_no_god_file_in_generation() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "generation"
    assert target.is_dir()
    budget = {"logic.py": 250, "parsers.py": 250, "__init__.py": 100}
    for path in target.glob("*.py"):
        with path.open() as fh:
            line_count = sum(1 for _ in fh)
        cap = budget.get(path.name, 250)
        assert line_count <= cap, f"{path.name} has {line_count} LOC > {cap}"


def test_parser_accepts_system_design() -> None:
    payload: dict[str, Any] = {
        "questions": [
            {
                "prompt_text": "Design a rate limiter for a distributed API.",
                "question_type": "system_design",
                "difficulty": "hard",
                "expected_depth": 4,
                "linked_outcome_id": None,
                "source_refs": [],
                "rationale": "architecture probe",
            }
        ]
    }
    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    assert parsed[0].question_type == "system_design"


def test_parser_assigns_server_group_ids_for_logical_question_indexes() -> None:
    outcome_id = uuid4()
    angles = ("technical", "system_design", "situational", "behavioral")
    payload = {
        "questions": [
            *[
                _question(
                    position=index,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                    logical_question_index=0,
                )
                for index, question_type in enumerate(angles, start=1)
            ],
            # A partial group (missing two angles) is discarded, not kept.
            _question(
                position=5,
                question_type="technical",
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=1,
            ),
            _question(
                position=6,
                question_type="system_design",
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=1,
            ),
        ]
    }

    parsed = parse_generation_response(payload, require_logical_question_index=True)

    assert len(parsed) == 4
    assert {draft.logical_question_index for draft in parsed} == {0}
    assert len({draft.variant_group_id for draft in parsed}) == 1
    assert all(question.variant_group_id is not None for question in parsed)


def test_all_angle_parser_discards_oversized_or_duplicate_groups() -> None:
    outcome_id = uuid4()
    payload = {
        "questions": [
            *[
                _question(
                    position=index,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                    logical_question_index=0,
                )
                for index, question_type in enumerate(
                    ("technical", "system_design", "situational", "behavioral", "technical"),
                    start=1,
                )
            ],
            *[
                _question(
                    position=10 + index,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                    logical_question_index=1,
                )
                for index, question_type in enumerate(
                    ("technical", "system_design", "situational", "behavioral"),
                )
            ],
        ]
    }

    parsed = parse_generation_response(payload, require_logical_question_index=True)

    assert len(parsed) == 4
    assert {draft.logical_question_index for draft in parsed} == {1}
    assert len({draft.variant_group_id for draft in parsed}) == 1


def test_all_angle_parser_caps_complete_interleaved_groups() -> None:
    outcome_id = uuid4()
    angles = ("technical", "system_design", "situational", "behavioral")
    payload = {
        "questions": [
            *[
                _question(
                    position=angle_index * 2 + group_index + 1,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                    logical_question_index=group_index,
                )
                for angle_index, question_type in enumerate(angles)
                for group_index in (0, 1)
            ],
            _question(
                position=9,
                question_type="technical",
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=2,
            ),
            _question(
                position=10,
                question_type="system_design",
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=2,
            ),
        ]
    }

    parsed = parse_generation_response(
        payload,
        max_questions=8,
        require_logical_question_index=True,
    )

    assert len(parsed) == 8
    by_group: dict[object, list[InterviewQuestionDraft]] = {}
    for draft in parsed:
        by_group.setdefault(draft.variant_group_id, []).append(draft)
    assert sorted(len(members) for members in by_group.values()) == [4, 4]
    assert all(
        len({member.question_type for member in members}) == 4 for members in by_group.values()
    )


def test_all_angle_parser_discards_partial_group_without_slicing() -> None:
    outcome_id = uuid4()
    payload = {
        "questions": [
            *[
                _question(
                    position=index,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                    logical_question_index=0,
                )
                for index, question_type in enumerate(
                    ("technical", "system_design", "situational", "behavioral"),
                )
            ],
            _question(
                position=5,
                question_type="technical",
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=1,
            ),
            _question(
                position=6,
                question_type="system_design",
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=1,
            ),
        ]
    }

    parsed = parse_generation_response(
        payload,
        max_questions=5,
        require_logical_question_index=True,
    )

    assert len(parsed) == 4
    assert {draft.logical_question_index for draft in parsed} == {0}


def test_all_angle_parser_rejects_all_rows_reusing_one_index() -> None:
    outcome_id = uuid4()
    payload = {
        "questions": [
            _question(
                position=index,
                question_type=("technical", "system_design", "situational", "behavioral")[
                    index % 4
                ],
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=0,
            )
            for index in range(8)
        ]
    }

    parsed = parse_generation_response(payload, require_logical_question_index=True)

    assert parsed == []


@pytest.mark.parametrize("ordinal", ["0", " 0 ", 0.0])
def test_all_angle_parser_accepts_stringified_ordinals(ordinal: object) -> None:
    outcome_id = uuid4()
    payload = {
        "questions": [
            {
                **_question(
                    position=index,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                ),
                "logical_question_index": ordinal,
            }
            for index, question_type in enumerate(
                ("technical", "system_design", "situational", "behavioral"),
                start=1,
            )
        ]
    }

    parsed = parse_generation_response(payload, require_logical_question_index=True)

    assert len(parsed) == 4
    assert {draft.logical_question_index for draft in parsed} == {0}
    assert len({draft.variant_group_id for draft in parsed}) == 1


@pytest.mark.parametrize(
    "ordinal",
    [True, -1, -1.0, "-1", "2.5", 2.5, "1e0", "abc", "", " ", float("nan"), float("inf")],
)
def test_all_angle_parser_rejects_invalid_ordinals(ordinal: object) -> None:
    outcome_id = uuid4()
    payload = {
        "questions": [
            {
                **_question(
                    position=index,
                    question_type=question_type,
                    difficulty="easy",
                    expected_depth=2,
                    linked_outcome_id=outcome_id,
                ),
                "logical_question_index": ordinal,
            }
            for index, question_type in enumerate(
                ("technical", "system_design", "situational", "behavioral"),
                start=1,
            )
        ]
    }

    assert parse_generation_response(payload, require_logical_question_index=True) == []


def test_resolve_variant_strategy() -> None:
    from abridgeai.features.interviews.ai.stages.generation.resolve import (
        resolve_variant_strategy,
    )

    assert resolve_variant_strategy({"variant_strategy": "all_angles"}) == "all_angles"
    assert resolve_variant_strategy({"variant_strategy": "role_only"}) == "role_only"
    assert resolve_variant_strategy({"variant_strategy": "ROLE_ONLY"}) == "role_only"
    assert resolve_variant_strategy({"variant_strategy": "bogus"}) is None
    assert resolve_variant_strategy({}) is None
    assert resolve_variant_strategy(None) is None


@pytest.mark.asyncio
async def test_all_angles_variant_mode_asks_for_logical_count() -> None:
    outcome_id = uuid4()
    payload = {
        "questions": [
            _question(
                position=index,
                question_type=question_type,
                difficulty="easy",
                expected_depth=2,
                linked_outcome_id=outcome_id,
                logical_question_index=0,
            )
            for index, question_type in enumerate(
                ("technical", "system_design", "situational", "behavioral"),
                start=1,
            )
        ]
    }
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(payload))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(question_count=8),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
        override_question_count=32,  # pipeline passes total = 8 logical x 4 angles
        variant_strategy="all_angles",
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Total LOGICAL questions to produce: 8" in user_prompt
    assert "system_design" in user_prompt
    assert "Total rows = 32" in user_prompt
    assert "logical_question_index" in user_prompt
    assert len(drafts) == 4
    assert len({draft.variant_group_id for draft in drafts}) == 1
    # Grounding rule (Slice 21 fix): the prompt must demand source_refs so the
    # LLM stops emitting empty arrays that the GROUNDED check then rejects.
    assert "MUST cite at least one chunk" in user_prompt


@pytest.mark.asyncio
async def test_role_only_variant_mode_fixes_type() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(question_count=8),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
        variant_strategy="role_only",
        role_type="technical",
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert 'Every question MUST have question_type "technical"' in user_prompt
