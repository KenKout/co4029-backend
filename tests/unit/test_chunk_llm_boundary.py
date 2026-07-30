"""Stage B' — LLM chunk-boundary decision.

The tests that matter here are the ones about what happens when the model is
wrong. A boundary decision is applied to the artefact every downstream stage
treats as ground truth, so the invariant is: whatever the model returns, the
concatenated output must reproduce the input document exactly, or the run must
fall back to one chunk per window.
"""

from __future__ import annotations

import pytest

from abridgeai.ai.chunking._llm_boundary import (
    group_by_llm_boundaries,
    split_on_barriers,
    validate_groups,
)
from abridgeai.ai.chunking.base import RawChunk


def _chunk(index: int, *, topic: int = 1, role: str = "body", tokens: int = 100) -> RawChunk:
    return RawChunk(
        content=f"content of window {index}",
        chunk_index=index,
        metadata={
            "topic_group_id": topic,
            "content_role": role,
            "token_count": tokens,
            "slide_title": f"Slide {index}",
        },
    )


class _FakeResult:
    def __init__(self, payload: object) -> None:
        self.content_json = payload


class _FakeGateway:
    """Returns queued payloads, one per call, recording the prompts it saw."""

    def __init__(self, payloads: list[object]) -> None:
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    async def generate_json(self, **kwargs: object) -> _FakeResult:
        self.prompts.append(str(kwargs.get("user_prompt")))
        return _FakeResult(self._payloads.pop(0) if self._payloads else None)


class TestBarriers:
    def test_topic_group_change_splits_a_run(self) -> None:
        chunks = [_chunk(0, topic=1), _chunk(1, topic=1), _chunk(2, topic=2)]
        assert split_on_barriers(chunks) == [[0, 1], [2]]

    def test_role_change_splits_a_run(self) -> None:
        chunks = [_chunk(0, role="body"), _chunk(1, role="summary")]
        assert split_on_barriers(chunks) == [[0], [1]]

    def test_runs_always_partition_the_document(self) -> None:
        chunks = [
            _chunk(0, topic=1),
            _chunk(1, topic=2),
            _chunk(2, topic=2),
            _chunk(3, topic=2, role="review"),
            _chunk(4, topic=3),
        ]
        runs = split_on_barriers(chunks)
        assert [i for run in runs for i in run] == [0, 1, 2, 3, 4]

    def test_a_repeated_topic_id_after_a_gap_does_not_rejoin(self) -> None:
        """Non-adjacent runs sharing an id are still separate runs.

        Merging them would pull together slides with other material between
        them, reordering the document.
        """
        chunks = [_chunk(0, topic=1), _chunk(1, topic=2), _chunk(2, topic=1)]
        assert split_on_barriers(chunks) == [[0], [1], [2]]


class TestValidateGroups:
    def _tokens(self, n: int, each: int = 100) -> dict[int, int]:
        return dict.fromkeys(range(n), each)

    def test_accepts_a_clean_partition(self) -> None:
        got = validate_groups(
            {"groups": [[0, 1], [2]]},
            expected=[0, 1, 2],
            tokens=self._tokens(3),
            max_window_tokens=2000,
        )
        assert got == [[0, 1], [2]]

    def test_rejects_a_missing_index(self) -> None:
        assert (
            validate_groups(
                {"groups": [[0, 1]]},
                expected=[0, 1, 2],
                tokens=self._tokens(3),
                max_window_tokens=2000,
            )
            is None
        )

    def test_rejects_a_duplicated_index(self) -> None:
        assert (
            validate_groups(
                {"groups": [[0, 1], [1, 2]]},
                expected=[0, 1, 2],
                tokens=self._tokens(3),
                max_window_tokens=2000,
            )
            is None
        )

    def test_rejects_a_non_consecutive_group(self) -> None:
        assert (
            validate_groups(
                {"groups": [[0, 2], [1]]},
                expected=[0, 1, 2],
                tokens=self._tokens(3),
                max_window_tokens=2000,
            )
            is None
        )

    def test_rejects_a_group_over_the_token_ceiling(self) -> None:
        assert (
            validate_groups(
                {"groups": [[0, 1, 2]]},
                expected=[0, 1, 2],
                tokens=self._tokens(3, each=900),
                max_window_tokens=2000,
            )
            is None
        )

    @pytest.mark.parametrize(
        "payload",
        [None, [], {}, {"groups": "nope"}, {"groups": [[]]}, {"groups": [["a"]]}],
    )
    def test_rejects_malformed_payloads(self, payload: object) -> None:
        assert (
            validate_groups(
                payload, expected=[0], tokens={0: 10}, max_window_tokens=2000
            )
            is None
        )


class TestGrouping:
    async def test_merges_within_a_topic_group(self) -> None:
        chunks = [_chunk(i, topic=1) for i in range(3)]
        gateway = _FakeGateway([{"groups": [[0, 1], [2]]}])
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object()
        )
        assert groups == [[0, 1], [2]]

    async def test_singleton_runs_cost_no_llm_call(self) -> None:
        chunks = [_chunk(0, topic=1), _chunk(1, topic=2), _chunk(2, topic=3)]
        gateway = _FakeGateway([])
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object()
        )
        assert groups == [[0], [1], [2]]
        assert gateway.prompts == []

    async def test_prompt_carries_digests_not_full_text(self) -> None:
        chunks = [_chunk(i, topic=1) for i in range(2)]
        gateway = _FakeGateway([{"groups": [[0, 1]]}])
        await group_by_llm_boundaries(chunks, llm_gateway=gateway, db=object())
        prompt = gateway.prompts[0]
        assert "[0]" in prompt
        assert "[1]" in prompt
        assert "100 tokens" in prompt
        assert "Slide 0" in prompt

    async def test_bad_response_falls_back_to_one_chunk_per_window(self) -> None:
        chunks = [_chunk(i, topic=1) for i in range(3)]
        gateway = _FakeGateway([{"groups": [[0, 2]]}])  # drops index 1
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object()
        )
        assert groups == [[0], [1], [2]]

    async def test_returns_none_without_a_gateway(self) -> None:
        """``None`` hands the decision back so the caller uses the glue path."""
        chunks = [_chunk(0)]
        assert await group_by_llm_boundaries(chunks, llm_gateway=None, db=object()) is None
        assert await group_by_llm_boundaries(chunks, llm_gateway=object(), db=None) is None

    async def test_output_always_reproduces_the_document(self) -> None:
        chunks = [_chunk(i, topic=1) for i in range(6)]
        gateway = _FakeGateway([{"groups": [[0, 1], [2], [3, 4, 5]]}])
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object()
        )
        assert groups is not None
        assert [i for g in groups for i in g] == list(range(6))


class TestCarryOver:
    async def test_trailing_group_is_re_shown_in_the_next_batch(self) -> None:
        """The last group of a batch is provisional until the model sees what follows.

        With batch_size=4 over 6 windows, the first call returns [[0,1],[2,3]].
        [2,3] is withheld — the model could not see window 4 — so the second
        call starts at index 2, not 4.
        """
        chunks = [_chunk(i, topic=1) for i in range(6)]
        gateway = _FakeGateway(
            [{"groups": [[0, 1], [2, 3]]}, {"groups": [[2, 3, 4], [5]]}]
        )
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object(), batch_size=4
        )
        assert groups == [[0, 1], [2, 3, 4], [5]]
        assert "indices 0-3" in gateway.prompts[0]
        assert "indices 2-5" in gateway.prompts[1]

    async def test_a_single_group_batch_still_terminates(self) -> None:
        """The infinite-loop case: the model calls the whole batch one group.

        Carrying it whole would advance the cursor by zero. ``max_carry`` forces
        it to be committed instead, so the run finishes.
        """
        chunks = [_chunk(i, topic=1, tokens=10) for i in range(8)]
        gateway = _FakeGateway([{"groups": [[0, 1, 2, 3]]}, {"groups": [[4, 5, 6, 7]]}])
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object(), batch_size=4, max_carry=2
        )
        assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert len(gateway.prompts) == 2


class _FakeCache:
    """Minimal ``ChunkingCache`` stand-in recording gets and puts."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, object]] = {}
        self.gets = 0

    async def get(self, content_hash: str, prompt_version: str) -> dict[str, object] | None:
        self.gets += 1
        row = self.rows.get((content_hash, prompt_version))
        return {"output_json": row} if row is not None else None

    async def put(
        self, content_hash: str, prompt_version: str, *, output_json: dict[str, object], **_: object
    ) -> None:
        self.rows[(content_hash, prompt_version)] = output_json


class TestBoundaryCache:
    async def test_second_run_over_identical_input_costs_no_call(self) -> None:
        chunks = [_chunk(i, topic=1) for i in range(3)]
        cache = _FakeCache()

        first = _FakeGateway([{"groups": [[0, 1], [2]]}])
        groups_a = await group_by_llm_boundaries(
            chunks, llm_gateway=first, db=object(), cache=cache
        )

        second = _FakeGateway([])  # would raise IndexError if called
        groups_b = await group_by_llm_boundaries(
            chunks, llm_gateway=second, db=object(), cache=cache
        )

        assert groups_a == groups_b == [[0, 1], [2]]
        assert second.prompts == []

    async def test_cache_key_follows_content_not_position(self) -> None:
        """The same run of slides re-ingested at a different offset still hits."""
        cache = _FakeCache()
        run = [_chunk(i, topic=1) for i in range(2)]
        gateway = _FakeGateway([{"groups": [[0, 1]]}])
        await group_by_llm_boundaries(run, llm_gateway=gateway, db=object(), cache=cache)

        # Same two windows, now preceded by an unrelated topic group.
        shifted = [_chunk(9, topic=7), *run]
        second = _FakeGateway([])
        groups = await group_by_llm_boundaries(
            shifted, llm_gateway=second, db=object(), cache=cache
        )
        assert groups == [[0], [1, 2]]
        assert second.prompts == []

    async def test_stale_cache_entry_is_ignored_not_applied(self) -> None:
        chunks = [_chunk(i, topic=1) for i in range(3)]
        cache = _FakeCache()
        gateway = _FakeGateway([{"groups": [[0, 1], [2]]}])
        await group_by_llm_boundaries(chunks, llm_gateway=gateway, db=object(), cache=cache)

        # Corrupt the stored partition so it no longer covers the run.
        key = next(iter(cache.rows))
        cache.rows[key] = {"groups": [[0]]}

        refetch = _FakeGateway([{"groups": [[0], [1], [2]]}])
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=refetch, db=object(), cache=cache
        )
        assert groups == [[0], [1], [2]]
        assert len(refetch.prompts) == 1

    async def test_terminates_when_carry_cap_exceeds_batch_size(self) -> None:
        """A misconfiguration must not hang the ingest.

        ``max_carry >= batch_size`` means the cap alone can never force a
        whole-batch group to be committed, so a separate guard has to.
        """
        chunks = [_chunk(i, topic=1, tokens=10) for i in range(6)]
        gateway = _FakeGateway(
            [{"groups": [[0, 1, 2]]}, {"groups": [[3, 4, 5]]}]
        )
        groups = await group_by_llm_boundaries(
            chunks, llm_gateway=gateway, db=object(), batch_size=3, max_carry=99
        )
        assert groups == [[0, 1, 2], [3, 4, 5]]
