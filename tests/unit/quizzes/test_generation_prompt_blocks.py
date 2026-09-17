"""The two blocks the generation prompt is built out of.

Neither of these can fail loudly. They render text into a prompt, the
model reads whatever arrives, and a mistake comes back as questions that
are slightly worse -- ungrounded, or ignoring a prerequisite ordering the
graph knew about. There is no exception and no wrong status code, which is
exactly why they are worth pinning.

**The chunk block** labels each source with ``[chunk_id]``. Those labels
are what the model cites in ``source_refs``, and the groundedness
validator downstream matches on them, so a chunk rendered without its id
is a chunk the model cannot attribute a question to.

**The knowledge-graph block** is the only place the model is told that one
concept must be learned before another. It is plain prose rather than
JSON, so its shape is its meaning: the arrow direction carries the
prerequisite relation, and a missing definition has to say so rather than
render as an empty tail that reads like the concept has no content.

Pure string rendering -- no gateway, no database, no prompt templates.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from abridgeai.ai.knowledge_graph.schemas import Concept, ConceptRelationship, KGContext
from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.features.quizzes.ai.stages.generation.logic import (
    _render_chunks,
    _render_kg_section,
)


def _chunk(content: str, chunk_id=None) -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=chunk_id or uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=content,
        distance=0.1,
    )


class TestTheSourceChunkBlock:
    def test_each_chunk_is_labelled_with_its_id(self) -> None:
        """The label is the model's only handle on a source.

        It cites these ids in ``source_refs``, and the validation stage
        checks the claim against the chunk they name -- so an unlabelled
        chunk is one no question can be grounded in.
        """
        chunk = _chunk("Photosynthesis converts light into chemical energy.")

        rendered = _render_chunks([chunk])

        assert f"[{chunk.chunk_id}]" in rendered
        assert "text: Photosynthesis converts light" in rendered

    def test_chunks_are_separated_so_they_do_not_read_as_one_source(self) -> None:
        """Run together, two chunks look like one longer passage and the
        model attributes a question spanning both to whichever id it saw
        first."""
        first, second = _chunk("First passage."), _chunk("Second passage.")

        rendered = _render_chunks([first, second])

        assert "\n\n" in rendered
        assert rendered.index(str(first.chunk_id)) < rendered.index(str(second.chunk_id))

    def test_the_caller_order_is_preserved(self) -> None:
        """Retrieval ranked these by relevance and rerank re-ordered them;
        the prompt must not undo that work."""
        chunks = [_chunk(f"Passage {n}") for n in range(4)]

        rendered = _render_chunks(chunks)

        positions = [rendered.index(str(c.chunk_id)) for c in chunks]
        assert positions == sorted(positions)

    def test_an_empty_pool_says_so_in_words(self) -> None:
        """An empty block would leave the model to invent questions from
        nothing while the prompt still asks for cited sources. Saying it
        outright is what lets the model produce nothing instead.
        """
        assert _render_chunks([]) == "No indexed chunks were found."

    def test_a_chunk_with_empty_content_still_carries_its_label(self) -> None:
        """An empty chunk is a retrieval or ingest problem, and dropping it
        here would hide that behind a shorter prompt."""
        chunk = _chunk("")
        assert f"[{chunk.chunk_id}]" in _render_chunks([chunk])


class TestTheKnowledgeGraphBlock:
    def test_the_section_is_empty_when_the_graph_is_off(self) -> None:
        """Not a placeholder or a heading with nothing under it: an empty
        string is what keeps the section out of the prompt entirely."""
        assert _render_kg_section(None) == ""

    def test_an_empty_graph_renders_nothing(self) -> None:
        """The feature is on but the lesson has no extracted concepts yet.

        A heading with no content below it reads to the model as "these
        are the concepts: none", which is worse than saying nothing.
        """
        assert _render_kg_section(KGContext(enabled=True)) == ""

    def test_concepts_are_listed_with_their_definitions(self) -> None:
        context = KGContext(
            concepts=[
                Concept(name="Photosynthesis", definition="Light to chemical energy."),
                Concept(name="Chlorophyll", definition="The pigment that absorbs light."),
            ],
            enabled=True,
        )

        rendered = _render_kg_section(context)

        assert "Knowledge graph concepts" in rendered
        assert "- Photosynthesis: Light to chemical energy." in rendered
        assert "- Chlorophyll: The pigment that absorbs light." in rendered

    def test_a_concept_with_no_definition_says_so(self) -> None:
        """Extraction records a name it could not define.

        Rendered as ``- Photosynthesis: `` the line reads like an empty
        definition, and the model has been observed filling that silence
        itself. The explicit marker is a statement rather than a gap.
        """
        context = KGContext(concepts=[Concept(name="Photosynthesis")], enabled=True)

        rendered = _render_kg_section(context)

        assert "- Photosynthesis: (no definition recorded)" in rendered

    def test_prerequisites_state_their_direction(self) -> None:
        """The arrow is the relation.

        Reversed, the model is told the dependent concept comes first and
        writes questions that assume knowledge the learner has not reached
        yet -- which is precisely the failure the graph exists to prevent.
        """
        context = KGContext(
            concepts=[Concept(name="Photosynthesis")],
            prerequisites=[
                ConceptRelationship(
                    source="Cell structure",
                    target="Photosynthesis",
                    relation="PREREQUISITE_OF",
                )
            ],
            enabled=True,
        )

        rendered = _render_kg_section(context)

        assert "source must be learned before target" in rendered
        assert "- Cell structure → Photosynthesis" in rendered

    def test_a_prerequisite_carries_its_evidence_when_there_is_any(self) -> None:
        """The evidence is the sentence the extractor based the edge on;
        it is what lets the model judge how much to lean on the claim."""
        context = KGContext(
            prerequisites=[
                ConceptRelationship(
                    source="Cell structure",
                    target="Photosynthesis",
                    evidence="Chapter 2 introduces the chloroplast first.",
                )
            ],
            enabled=True,
        )

        rendered = _render_kg_section(context)

        assert "Cell structure → Photosynthesis — Chapter 2 introduces" in rendered

    def test_a_prerequisite_without_evidence_has_no_dangling_dash(self) -> None:
        """A trailing em dash reads as a truncated line."""
        context = KGContext(
            prerequisites=[ConceptRelationship(source="A", target="B")], enabled=True
        )

        rendered = _render_kg_section(context)

        assert "- A → B" in rendered
        assert "A → B —" not in rendered

    def test_related_concepts_use_a_symmetric_arrow(self) -> None:
        """A different arrow because the relation is different: related is
        undirected, and rendering it like a prerequisite would invent an
        ordering the graph never asserted.
        """
        context = KGContext(
            related=[ConceptRelationship(source="Respiration", target="Photosynthesis")],
            enabled=True,
        )

        rendered = _render_kg_section(context)

        assert "Related concepts:" in rendered
        assert "- Respiration ↔ Photosynthesis" in rendered

    def test_the_three_sections_are_separated_by_blank_lines(self) -> None:
        """They are three different claims about the lesson; run together
        the model reads the prerequisite list as more concepts."""
        context = KGContext(
            concepts=[Concept(name="Photosynthesis", definition="d")],
            prerequisites=[ConceptRelationship(source="A", target="B")],
            related=[ConceptRelationship(source="C", target="D")],
            enabled=True,
        )

        rendered = _render_kg_section(context)

        assert "\n\nPrerequisite chains" in rendered
        assert "\n\nRelated concepts:" in rendered

    @pytest.mark.parametrize(
        ("field", "expected_heading", "absent_heading"),
        [
            ("prerequisites", "Prerequisite chains", "Related concepts"),
            ("related", "Related concepts", "Prerequisite chains"),
        ],
    )
    def test_an_absent_relation_kind_gets_no_heading(
        self, field: str, expected_heading: str, absent_heading: str
    ) -> None:
        """A lesson can have prerequisites and no related pairs, or the
        reverse. An empty heading tells the model a category exists and is
        empty, which is not what the absence means.
        """
        context = KGContext(
            **{field: [ConceptRelationship(source="A", target="B")]}, enabled=True
        )

        rendered = _render_kg_section(context)

        assert expected_heading in rendered
        assert absent_heading not in rendered

    def test_a_graph_of_only_relations_still_renders(self) -> None:
        """``is_empty`` is false when there are edges but no concepts, so
        the concept heading appears with nothing under it -- pinned as the
        current shape rather than asserted to be ideal.
        """
        context = KGContext(
            prerequisites=[ConceptRelationship(source="A", target="B")], enabled=True
        )

        rendered = _render_kg_section(context)

        assert rendered.startswith("Knowledge graph concepts")
        assert "- A → B" in rendered
