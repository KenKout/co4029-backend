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


def test_renumber_display_positions_survives_client_position_sort():
    """position must encode the SHUFFLED display slot, not the authored order.

    The SPA defensively re-sorts questions and options by ``position`` before
    rendering. If the take payload carried the authored position, that sort
    would undo the shuffle and leak the canonical order. After
    _renumber_display_positions the list is 1..N in its current (post-layout)
    order, so a stable sort by position is a no-op and the shuffle survives.
    """
    from types import SimpleNamespace

    from abridgeai.features.quizzes.services.taking import _renumber_display_positions

    # Simulate a post-apply_layout order where authored positions are jumbled:
    # authored positions [3, 1, 2] in display order; options likewise reversed.
    questions = [
        SimpleNamespace(
            position=3,
            options=[SimpleNamespace(position=2), SimpleNamespace(position=1)],
        ),
        SimpleNamespace(position=1, options=[SimpleNamespace(position=5)]),
        SimpleNamespace(position=2, options=[]),
    ]

    _renumber_display_positions(questions)  # type: ignore[arg-type]

    assert [q.position for q in questions] == [1, 2, 3]
    # Sorting by the new position preserves the display order (no-op).
    resorted = sorted(questions, key=lambda q: q.position)
    assert resorted == questions
    # Options renumbered 1..N within each question, in their given order.
    assert [o.position for o in questions[0].options] == [1, 2]
    assert [o.position for o in questions[1].options] == [1]
