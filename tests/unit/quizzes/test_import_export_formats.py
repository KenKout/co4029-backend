"""Phase 11: GIFT + Moodle-XML parser/serializer unit tests (no DB)."""

from __future__ import annotations

import pytest

from abridgeai.features.quizzes.services.formats.gift import (
    GiftParseError,
    parse_gift,
    serialize_gift,
)
from abridgeai.features.quizzes.services.formats.moodle_xml import (
    XmlParseError,
    parse_moodle_xml,
    serialize_moodle_xml,
)

GIFT = """::Q1:: What is 2+2? {=4 ~3 ~5}

Sky is blue.{TRUE}

What colour? {=blue}
"""

XML = """<?xml version="1.0"?>
<quiz>
  <question type="multichoice">
    <questiontext format="html"><text><![CDATA[What is 2+2?]]></text></questiontext>
    <answer fraction="100"><text>4</text></answer>
    <answer fraction="0"><text>3</text></answer>
  </question>
  <question type="truefalse">
    <questiontext><text>Sky is blue.</text></questiontext>
    <answer fraction="100"><text>true</text></answer>
    <answer fraction="0"><text>false</text></answer>
  </question>
  <question type="category">
    <category><text>$course$/Imported</text></category>
  </question>
</quiz>"""


def test_gift_parses_mcq_tf_shortanswer():
    res = parse_gift(GIFT)
    assert len(res.questions) == 3
    assert res.questions[0].question_type == "multiple_choice"
    assert res.questions[1].question_type == "true_false"
    assert res.questions[2].question_type == "short_answer"
    assert res.questions[2].correct_answer == "blue"


def test_gift_malformed_raises():
    with pytest.raises(GiftParseError):
        parse_gift("this block has no answer braces")


def test_gift_roundtrip_preserves_question_count():
    res = parse_gift(GIFT)
    out = serialize_gift(res.questions)
    reparsed = parse_gift(out)
    assert len(reparsed.questions) == len(res.questions)


def test_xml_parses_and_skips_category():
    res = parse_moodle_xml(XML)
    assert len(res.questions) == 2  # 'category' pseudo-question skipped
    assert res.questions[0].question_type == "multiple_choice"
    assert res.questions[0].prompt_text == "What is 2+2?"
    assert res.questions[1].question_type == "true_false"


def test_xml_malformed_raises():
    with pytest.raises(XmlParseError):
        parse_moodle_xml("<quiz><question></quiz>")


def test_xml_roundtrip_preserves_question_count():
    res = parse_moodle_xml(XML)
    out = serialize_moodle_xml(res.questions)
    reparsed = parse_moodle_xml(out)
    assert len(reparsed.questions) == len(res.questions)
