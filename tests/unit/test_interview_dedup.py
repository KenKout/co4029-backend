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
    return ShortlistedQuestion(question_id=uuid.uuid4(), prompt_text=text, distance=distance)


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


# ── batch embedding persistence (the AI-generated-bank regression) ────────────
#
# `interview_questions.embedding` was populated only by `add_question` /
# `update_question`. The generation pipeline inserted rows directly, so every
# AI-generated question kept `embedding = NULL`, and because the shortlist
# filters on `embedding IS NOT NULL` those rows were invisible to the checker:
# duplicate detection silently answered "not a duplicate" on any bank nobody had
# hand-edited. These pin the shared helper the pipeline now calls.


class _RecordingDb:
    """Captures the UPDATE statements a caller issues."""

    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    async def execute(self, sql: object, params: dict[str, object] | None = None) -> object:
        self.updates.append({"sql": str(sql), "params": params or {}})

        class _Result:
            def all(self) -> list[object]:
                return []

        return _Result()


class _FakeEmbeddingClient:
    """Returns one deterministic vector per input and counts its calls."""

    def __init__(self, *, fail: bool = False, dim: int = 3) -> None:
        self.fail = fail
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("provider down")
        return [[float(i)] * self.dim for i, _ in enumerate(texts)]


@pytest.fixture
def dedup_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the feature flag on regardless of the ambient .env."""
    from abridgeai.features.interviews import dedup as dedup_mod

    class _S:
        interview_dedup_enabled = True

    monkeypatch.setattr(dedup_mod, "get_settings", lambda: _S())


@pytest.mark.asyncio
@pytest.mark.usefixtures("dedup_on")
async def test_store_embeddings_uses_one_provider_call_for_the_whole_batch() -> None:
    """A generation run of N questions must not cost N round trips."""
    from abridgeai.features.interviews.dedup import store_question_embeddings

    db = _RecordingDb()
    client = _FakeEmbeddingClient()
    ids = [uuid.uuid4() for _ in range(4)]

    stored = await store_question_embeddings(
        db,  # type: ignore[arg-type]
        question_ids=ids,
        prompt_texts=["q1", "q2", "q3", "q4"],
        embedding_client=client,  # type: ignore[arg-type]
    )

    assert stored == 4
    assert len(client.calls) == 1, "batched into a single embed() call"
    assert client.calls[0] == ["q1", "q2", "q3", "q4"]
    assert len(db.updates) == 4
    assert {u["params"]["question_id"] for u in db.updates} == set(ids)  # type: ignore[index]
    # pgvector text format, cast to the column's halfvec type.
    assert str(db.updates[0]["params"]["embedding"]).startswith("[")  # type: ignore[index]
    assert "CAST(:embedding AS halfvec)" in str(db.updates[0]["sql"])


@pytest.mark.asyncio
async def test_store_embeddings_is_a_noop_when_the_feature_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No embedding spend for a feature nobody turned on."""
    from abridgeai.features.interviews import dedup as dedup_mod

    class _S:
        interview_dedup_enabled = False

    monkeypatch.setattr(dedup_mod, "get_settings", lambda: _S())

    db = _RecordingDb()
    client = _FakeEmbeddingClient()
    stored = await dedup_mod.store_question_embeddings(
        db,  # type: ignore[arg-type]
        question_ids=[uuid.uuid4()],
        prompt_texts=["q"],
        embedding_client=client,  # type: ignore[arg-type]
    )
    assert stored == 0
    assert client.calls == []
    assert db.updates == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("dedup_on")
async def test_store_embeddings_fails_open_on_a_provider_error() -> None:
    """A dead provider must not abort a completed generation run."""
    from abridgeai.features.interviews.dedup import store_question_embeddings

    db = _RecordingDb()
    stored = await store_question_embeddings(
        db,  # type: ignore[arg-type]
        question_ids=[uuid.uuid4()],
        prompt_texts=["q"],
        embedding_client=_FakeEmbeddingClient(fail=True),  # type: ignore[arg-type]
    )
    assert stored == 0
    assert db.updates == [], "no partial write when the embed call raised"


@pytest.mark.asyncio
@pytest.mark.usefixtures("dedup_on")
async def test_store_embeddings_skips_blank_prompts_without_calling_the_provider() -> None:
    from abridgeai.features.interviews.dedup import store_question_embeddings

    db = _RecordingDb()
    client = _FakeEmbeddingClient()
    keep = uuid.uuid4()

    stored = await store_question_embeddings(
        db,  # type: ignore[arg-type]
        question_ids=[uuid.uuid4(), keep, uuid.uuid4()],
        prompt_texts=["   ", "real question", ""],
        embedding_client=client,  # type: ignore[arg-type]
    )

    assert stored == 1
    assert client.calls == [["real question"]]
    assert db.updates[0]["params"]["question_id"] == keep  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.usefixtures("dedup_on")
async def test_store_embeddings_rejects_mismatched_input_lengths() -> None:
    """A zip() bug here would silently attach a vector to the wrong question."""
    from abridgeai.features.interviews.dedup import store_question_embeddings

    with pytest.raises(ValueError, match="same length"):
        await store_question_embeddings(
            _RecordingDb(),  # type: ignore[arg-type]
            question_ids=[uuid.uuid4(), uuid.uuid4()],
            prompt_texts=["only one"],
            embedding_client=_FakeEmbeddingClient(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.usefixtures("dedup_on")
async def test_store_embeddings_does_nothing_for_an_empty_batch() -> None:
    """A generation run that persisted zero questions must not call the provider."""
    from abridgeai.features.interviews.dedup import store_question_embeddings

    client = _FakeEmbeddingClient()
    assert (
        await store_question_embeddings(
            _RecordingDb(),  # type: ignore[arg-type]
            question_ids=[],
            prompt_texts=[],
            embedding_client=client,  # type: ignore[arg-type]
        )
        == 0
    )
    assert client.calls == []


# ── the pipeline actually calls it ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generation_pipeline_embeds_the_questions_it_persists() -> None:
    """The regression guard: `_persist_questions` must embed what it inserted.

    Asserting on the helper alone would not have caught the original bug — the
    helper did not exist and the pipeline simply never embedded. This pins the
    call, the ids/prompts handed over, and that the run id is threaded through so
    the embedding call lands under the run's `ai_pipeline_runs` parent.
    """
    from unittest.mock import AsyncMock, patch

    from abridgeai.features.interviews.ai.pipelines import (
        generation as gen,
    )
    from abridgeai.features.interviews.ai.pipelines import (
        persistence as pers,
    )

    class _Draft:
        def __init__(self, prompt: str) -> None:
            self.prompt_text = prompt
            self.linked_outcome_id = None
            self.variant_group_id = None
            self.question_type = "conceptual"
            self.difficulty = None
            self.model_answer = ""
            self.source_refs: list[object] = []

    class _Config:
        id = uuid.uuid4()
        module_id = uuid.uuid4()

    added: list[object] = []

    class _Db:
        def add(self, obj: object) -> None:
            # Mimic the DB assigning a PK on flush.
            obj.id = uuid.uuid4()  # type: ignore[attr-defined]
            added.append(obj)

        async def flush(self) -> None:
            return None

    run_id = uuid.uuid4()
    drafts = [_Draft("first question"), _Draft("second question")]

    with (
        patch.object(pers, "next_question_position", AsyncMock(side_effect=[1, 2])),
        patch.object(pers, "store_question_embeddings", AsyncMock(return_value=2)) as spy,
        patch(pers.__name__ + ".lock_question_append", AsyncMock()),
    ):
        await gen._persist_questions(
            _Db(),  # type: ignore[arg-type]
            config=_Config(),  # type: ignore[arg-type]
            accepted=drafts,  # type: ignore[arg-type]
            source_module_ids=["m1"],
            pipeline_run_id=run_id,
        )

    assert len(added) == 2, "both drafts persisted"
    spy.assert_awaited_once()
    kwargs = spy.await_args.kwargs
    assert kwargs["prompt_texts"] == ["first question", "second question"]
    assert kwargs["question_ids"] == [obj.id for obj in added]
    assert kwargs["pipeline_run_id"] == run_id


@pytest.mark.asyncio
async def test_generation_pipeline_survives_an_embedding_failure() -> None:
    """A provider outage must not fail a run whose questions are otherwise fine."""
    from unittest.mock import AsyncMock, patch

    from abridgeai.features.interviews.ai.pipelines import (
        generation as gen,
    )
    from abridgeai.features.interviews.ai.pipelines import (
        persistence as pers,
    )

    class _Draft:
        prompt_text = "q"
        linked_outcome_id = None
        variant_group_id = None
        question_type = "conceptual"
        difficulty = None
        model_answer = ""
        source_refs: list[object] = []

    class _Config:
        id = uuid.uuid4()
        module_id = uuid.uuid4()

    class _Db:
        def add(self, obj: object) -> None:
            obj.id = uuid.uuid4()  # type: ignore[attr-defined]

        async def flush(self) -> None:
            return None

    # store_question_embeddings swallows internally and returns 0; assert the
    # pipeline treats that as a non-event rather than propagating.
    with (
        patch.object(pers, "next_question_position", AsyncMock(return_value=1)),
        patch.object(pers, "store_question_embeddings", AsyncMock(return_value=0)),
        patch(pers.__name__ + ".lock_question_append", AsyncMock()),
    ):
        await gen._persist_questions(
            _Db(),  # type: ignore[arg-type]
            config=_Config(),  # type: ignore[arg-type]
            accepted=[_Draft()],  # type: ignore[arg-type]
            source_module_ids=[],
            pipeline_run_id=uuid.uuid4(),
        )
