"""Unit tests for interview question-bank duplicate detection.

Covers the pure layers: verdict parsing (every fail-open path) and the
shortlist's cosine-distance handling. The pgvector query and the LLM call are
I/O and are exercised via injected fakes, not against a live provider.

The behaviour these pin down is the fail-open contract: a broken judge must look
like "check unavailable", never like "confirmed unique", because a teacher acts
differently on those two.
"""

from __future__ import annotations

import uuid

import pytest

from abridgeai.ai.llm import LLMRole
from abridgeai.ai.llm.roles import INTERACTIVE_LLM_ROLES
from abridgeai.features.interviews.dedup import (
    MAX_SHORTLIST_DISTANCE,
    SHORTLIST_SIZE,
    DuplicateVerdict,
    ShortlistedQuestion,
    judge_duplicate,
    parse_duplicate_verdict,
    shortlist_similar_questions,
)


def _candidate(text: str = "What is an index?", distance: float = 0.1) -> ShortlistedQuestion:
    return ShortlistedQuestion(
        question_id=uuid.uuid4(), prompt_text=text, distance=distance
    )


# ── verdict parsing ──────────────────────────────────────────────────────────


def test_parses_a_confirmed_duplicate() -> None:
    candidates = [_candidate("A?"), _candidate("B?")]
    verdict = parse_duplicate_verdict(
        {"is_duplicate": True, "duplicate_of_index": 1, "rationale": "same ask"},
        candidates=candidates,
    )
    assert verdict.is_duplicate is True
    assert verdict.duplicate_of_id == candidates[1].question_id
    assert verdict.duplicate_of_text == "B?"
    assert verdict.rationale == "same ask"
    assert verdict.checked is True


def test_parses_a_clean_pass() -> None:
    verdict = parse_duplicate_verdict(
        {"is_duplicate": False, "duplicate_of_index": None, "rationale": "different angle"},
        candidates=[_candidate()],
    )
    assert verdict.is_duplicate is False
    assert verdict.duplicate_of_id is None
    assert verdict.checked is True  # a real judgement was made


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"is_duplicate": "yes"},  # string, not bool
        {"is_duplicate": 1},  # int, not bool
    ],
)
def test_malformed_payload_is_an_error_not_a_pass(payload: object) -> None:
    """The distinction that matters: unavailable != verified unique."""
    verdict = parse_duplicate_verdict(payload, candidates=[_candidate()])  # type: ignore[arg-type]
    assert verdict.is_duplicate is False
    assert verdict.checked is False
    assert verdict.error


def test_duplicate_without_index_is_an_error() -> None:
    """Which question it duplicates is the whole point — refuse to guess."""
    verdict = parse_duplicate_verdict(
        {"is_duplicate": True, "rationale": "feels similar"},
        candidates=[_candidate()],
    )
    assert verdict.is_duplicate is False
    assert verdict.checked is False


def test_true_is_not_accepted_as_index_one() -> None:
    """bool is an int subclass in Python; True must not select candidates[1]."""
    candidates = [_candidate("A?"), _candidate("B?")]
    verdict = parse_duplicate_verdict(
        {"is_duplicate": True, "duplicate_of_index": True}, candidates=candidates
    )
    assert verdict.is_duplicate is False
    assert verdict.checked is False


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_out_of_range_index_is_an_error(index: int) -> None:
    """Never attach a duplicate claim to the wrong question."""
    verdict = parse_duplicate_verdict(
        {"is_duplicate": True, "duplicate_of_index": index},
        candidates=[_candidate("A?"), _candidate("B?")],
    )
    assert verdict.is_duplicate is False
    assert verdict.checked is False
    assert "out of range" in verdict.error


def test_to_dict_is_json_safe() -> None:
    candidates = [_candidate("A?")]
    payload = parse_duplicate_verdict(
        {"is_duplicate": True, "duplicate_of_index": 0, "rationale": "r"},
        candidates=candidates,
    ).to_dict()
    assert payload["duplicate_of_id"] == str(candidates[0].question_id)
    assert isinstance(payload["is_duplicate"], bool)


# ── the judge's short-circuits ───────────────────────────────────────────────


class _ExplodingGateway:
    """Any call is a test failure — used to prove no LLM call happens."""

    async def generate_json(self, **_kwargs: object) -> object:  # pragma: no cover
        raise AssertionError("judge must not call the LLM here")


@pytest.mark.asyncio
async def test_empty_shortlist_costs_nothing() -> None:
    """A fresh bank has no candidates; that must not spend a call."""
    verdict = await judge_duplicate(
        db=None,  # type: ignore[arg-type]
        prompt_text="Anything?",
        candidates=[],
        gateway=_ExplodingGateway(),  # type: ignore[arg-type]
    )
    assert verdict.is_duplicate is False
    assert verdict.checked is True


@pytest.mark.asyncio
async def test_blank_proposal_costs_nothing() -> None:
    verdict = await judge_duplicate(
        db=None,  # type: ignore[arg-type]
        prompt_text="   ",
        candidates=[_candidate()],
        gateway=_ExplodingGateway(),  # type: ignore[arg-type]
    )
    assert verdict.is_duplicate is False


@pytest.mark.asyncio
async def test_gateway_failure_fails_open_with_an_error() -> None:
    class _Failing:
        async def generate_json(self, **_kwargs: object) -> object:
            raise RuntimeError("provider down")

    verdict = await judge_duplicate(
        db=None,  # type: ignore[arg-type]
        prompt_text="What is an index?",
        candidates=[_candidate()],
        gateway=_Failing(),  # type: ignore[arg-type]
    )
    # Fail OPEN: the teacher's save proceeds...
    assert verdict.is_duplicate is False
    # ...but the caller can tell the check did not actually run.
    assert verdict.checked is False
    assert "provider down" in verdict.error


# ── shortlist value semantics ────────────────────────────────────────────────


def test_similarity_is_the_complement_of_distance() -> None:
    assert _candidate(distance=0.0).similarity == pytest.approx(1.0)
    assert _candidate(distance=1.0).similarity == pytest.approx(0.0)


def test_shortlist_bounds_are_sane() -> None:
    # Small shortlist: everything goes in one prompt, so this bounds cost.
    assert 1 <= SHORTLIST_SIZE <= 10
    # Loose gate: the vector stage narrows, it must not be the decider.
    assert 0.0 < MAX_SHORTLIST_DISTANCE < 1.0


@pytest.mark.asyncio
async def test_shortlist_omits_the_exclude_param_when_there_is_nothing_to_exclude() -> None:
    """Regression: a bare NULL parameter breaks the query in Postgres.

    The original SQL used ``AND (:exclude_id IS NULL OR id <> :exclude_id)``. That
    is valid SQL but Postgres cannot infer a type for the NULL bind and rejects it
    with ``AmbiguousParameter`` — which only showed up against a real database, not
    against a mocked session. Assert on the emitted SQL/params so the shape stays
    fixed.
    """
    captured: dict[str, object] = {}

    class _Db:
        async def execute(self, sql: object, params: dict[str, object]) -> object:
            captured["sql"] = str(sql)
            captured["params"] = params

            class _Result:
                def all(self) -> list[object]:
                    return []

            return _Result()

    await shortlist_similar_questions(
        _Db(),  # type: ignore[arg-type]
        config_id=uuid.uuid4(),
        embedding=[0.1, 0.2, 0.3],
    )
    assert "exclude_id" not in captured["params"]  # type: ignore[operator]
    assert "exclude_id" not in str(captured["sql"])


@pytest.mark.asyncio
async def test_shortlist_binds_the_exclude_param_when_given_one() -> None:
    """Editing a question must not match the question against itself."""
    captured: dict[str, object] = {}
    own_id = uuid.uuid4()

    class _Db:
        async def execute(self, sql: object, params: dict[str, object]) -> object:
            captured["sql"] = str(sql)
            captured["params"] = params

            class _Result:
                def all(self) -> list[object]:
                    return []

            return _Result()

    await shortlist_similar_questions(
        _Db(),  # type: ignore[arg-type]
        config_id=uuid.uuid4(),
        embedding=[0.1, 0.2, 0.3],
        exclude_question_id=own_id,
    )
    assert captured["params"]["exclude_id"] == own_id  # type: ignore[index]
    assert "id <> :exclude_id" in str(captured["sql"])


@pytest.mark.asyncio
async def test_shortlist_skips_the_query_for_an_empty_embedding() -> None:
    class _Db:
        async def execute(self, *_a: object, **_k: object) -> object:  # pragma: no cover
            raise AssertionError("must not query without a vector")

    assert (
        await shortlist_similar_questions(
            _Db(),  # type: ignore[arg-type]
            config_id=uuid.uuid4(),
            embedding=[],
        )
        == []
    )


# ── role wiring ──────────────────────────────────────────────────────────────


def test_dedup_role_is_interactive() -> None:
    """A teacher is waiting on this, unlike the offline quality judge."""
    assert LLMRole.INTERVIEW_DEDUP in INTERACTIVE_LLM_ROLES
    assert LLMRole.INTERVIEW_QUALITY_JUDGE not in INTERACTIVE_LLM_ROLES


def test_not_duplicate_singleton_is_a_clean_pass() -> None:
    from abridgeai.features.interviews.dedup import NOT_DUPLICATE

    assert isinstance(NOT_DUPLICATE, DuplicateVerdict)
    assert NOT_DUPLICATE.is_duplicate is False
    assert NOT_DUPLICATE.checked is True
