from dataclasses import dataclass

from abridgeai.core.pagination import paginate_sequence


@dataclass(frozen=True)
class _Row:
    name: str
    score: int | None
    status: str


ROWS = [
    _Row("Alpha", 20, "passed"),
    _Row("Beta", None, "passed"),
    _Row("Gamma", 10, "failed"),
    _Row("Alphabet", 30, "passed"),
]


def test_sequence_search_filter_sort_and_page() -> None:
    result = paginate_sequence(
        ROWS,
        page=0,
        page_size=1,
        search="alpha",
        search_text=lambda row: row.name,
        predicate=lambda row: row.status == "passed",
        sort="score",
        sort_dir="desc",
        sortable={"score": lambda row: row.score},
    )

    assert result.items == [_Row("Alphabet", 30, "passed")]
    assert result.total == 2
    assert result.total_pages == 2


def test_sequence_keeps_missing_sort_values_last_in_both_directions() -> None:
    for direction in ("asc", "desc"):
        result = paginate_sequence(
            ROWS,
            page=0,
            page_size=20,
            sort="score",
            sort_dir=direction,
            sortable={"score": lambda row: row.score},
        )
        assert result.items[-1].score is None


def test_sequence_clamps_page_inputs() -> None:
    result = paginate_sequence(ROWS, page=-2, page_size=500)

    assert result.page == 0
    assert result.page_size == 200
    assert result.total == len(ROWS)
