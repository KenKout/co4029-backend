"""Concept consolidation grouping rules.

The Neo4j rewiring is exercised in ``tests/integration/test_kg_builder.py``;
what is tested here is the decision of *which* names denote one concept, which
is where the risk lives. Merging is destructive: an over-eager rule silently
rewrites what a course says, so the negative cases below matter more than the
positive ones.

The fixtures are real names taken from a single 34-slide lecture on
Management Support Systems, where per-chunk extraction produced 123 concepts
of which 14 were duplicate spellings.
"""

from __future__ import annotations

from abridgeai.ai.knowledge_graph.consolidate import (
    canonical_key,
    choose_canonical,
    plan_merges,
)


def _concept(
    name: str, *, mentions: int = 1, degree: int = 0, has_definition: bool = False
) -> dict[str, object]:
    return {
        "name": name,
        "mentions": mentions,
        "degree": degree,
        "has_definition": has_definition,
    }


class TestCanonicalKey:
    def test_plural_and_singular_agree(self) -> None:
        assert canonical_key("Expert system") == canonical_key("Expert systems")

    def test_parenthetical_acronym_is_dropped(self) -> None:
        assert canonical_key("Group support systems (GSS)") == canonical_key(
            "Group support systems"
        )

    def test_mid_phrase_parenthetical_is_dropped(self) -> None:
        assert canonical_key("Supply chain management (SCM) systems") == canonical_key(
            "Supply chain management system"
        )

    def test_curly_and_straight_apostrophes_agree(self) -> None:
        assert canonical_key("Simon’s decision-making process") == canonical_key(
            "Simon's decision-making process"
        )

    def test_case_and_punctuation_are_folded(self) -> None:
        assert canonical_key("DECISION-MAKING, PROCESS") == canonical_key(
            "decision making process"
        )

    def test_words_ending_in_s_keep_their_s(self) -> None:
        # The singulariser must not maul already-singular nouns.
        for word in ("process", "analysis", "business", "status"):
            assert canonical_key(word) == word

    def test_irregular_plurals_fold_consistently(self) -> None:
        assert canonical_key("technologies") == canonical_key("technology")
        assert canonical_key("processes") == canonical_key("process")
        assert canonical_key("warehouses") == canonical_key("warehouse")

    def test_distinct_concepts_keep_distinct_keys(self) -> None:
        # Head-noun stripping is deliberately NOT done: these stay apart.
        assert canonical_key("Decision support") != canonical_key(
            "Decision support systems"
        )
        assert canonical_key("Management") != canonical_key("Management control")


class TestPlanMerges:
    def test_groups_duplicate_spellings(self) -> None:
        groups = plan_merges(
            [
                _concept("Expert Systems"),
                _concept("Expert system"),
                _concept("Expert systems (ES)"),
            ]
        )
        assert len(groups) == 1
        assert len(next(iter(groups.values()))) == 3

    def test_singletons_are_not_returned(self) -> None:
        assert plan_merges([_concept("Data warehouse"), _concept("Star schema")]) == {}

    def test_bare_acronym_folds_into_its_expansion(self) -> None:
        groups = plan_merges(
            [_concept("Group support systems (GSS)"), _concept("GSS")]
        )
        assert len(groups) == 1
        names = {c["name"] for c in next(iter(groups.values()))}
        assert names == {"Group support systems (GSS)", "GSS"}

    def test_ambiguous_acronym_never_merges(self) -> None:
        """The case that makes a naive acronym map dangerous.

        One lecture used ``(EIS)`` for both Executive Information Systems and
        Enterprise Information Systems. Fusing them would merge two genuinely
        different concepts, so an acronym claimed by two expansions resolves
        to neither.
        """
        groups = plan_merges(
            [
                _concept("Executive Information System (EIS)"),
                _concept("Executive Information Systems"),
                _concept("Enterprise information systems (EIS)"),
                _concept("Enterprise Information Systems"),
                _concept("EIS"),
            ]
        )
        merged_names = [{c["name"] for c in members} for members in groups.values()]
        assert {"Executive Information System (EIS)", "Executive Information Systems"} in (
            merged_names
        )
        assert {
            "Enterprise information systems (EIS)",
            "Enterprise Information Systems",
        } in merged_names
        # The bare acronym resolves to neither expansion and stays alone.
        assert all("EIS" not in names or len(names) > 1 for names in merged_names)
        for names in merged_names:
            assert not (
                "Executive Information Systems" in names
                and "Enterprise Information Systems" in names
            )


class TestChooseCanonical:
    def test_most_mentioned_spelling_wins(self) -> None:
        winner = choose_canonical(
            [
                _concept("Decision Support System (DSS)", mentions=2, degree=30),
                _concept("Decision Support Systems", mentions=9, degree=20),
            ]
        )
        assert winner["name"] == "Decision Support Systems"

    def test_degree_breaks_a_mention_tie(self) -> None:
        winner = choose_canonical(
            [
                _concept("Expert system", mentions=3, degree=1),
                _concept("Expert Systems", mentions=3, degree=4),
            ]
        )
        assert winner["name"] == "Expert Systems"

    def test_definition_breaks_a_degree_tie(self) -> None:
        winner = choose_canonical(
            [
                _concept("Groupware", mentions=1, degree=1),
                _concept("Group ware", mentions=1, degree=1, has_definition=True),
            ]
        )
        assert winner["name"] == "Group ware"
