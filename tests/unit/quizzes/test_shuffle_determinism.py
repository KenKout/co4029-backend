"""Phase 6: deterministic per-attempt shuffle tests."""

from __future__ import annotations

import uuid

from abridgeai.features.quizzes.services.shuffle import build_layout

AID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def test_layout_is_reproducible_for_same_attempt():
    qids = [uuid.uuid4() for _ in range(6)]
    opts = {q: [uuid.uuid4() for _ in range(4)] for q in qids}
    a = build_layout(AID, qids, opts, shuffle_questions=True, shuffle_options=True)
    b = build_layout(AID, qids, opts, shuffle_questions=True, shuffle_options=True)
    assert a == b


def test_layout_differs_across_attempts():
    qids = [uuid.uuid4() for _ in range(8)]
    opts = {q: [] for q in qids}
    a = build_layout(AID, qids, opts, shuffle_questions=True, shuffle_options=False)
    other = uuid.UUID("87654321-4321-8765-4321-876543218765")
    b = build_layout(other, qids, opts, shuffle_questions=True, shuffle_options=False)
    # extremely unlikely to match across two different seeds on 8 items
    assert a["question_order"] != b["question_order"]


def test_no_shuffle_preserves_order():
    qids = [uuid.uuid4() for _ in range(5)]
    opts = {q: [] for q in qids}
    layout = build_layout(AID, qids, opts, shuffle_questions=False, shuffle_options=False)
    assert layout["question_order"] == [str(q) for q in qids]
