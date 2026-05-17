from __future__ import annotations

import json

import pytest

from abridgeai.ai.llm.gateway import _parse_llm_json


class TestParseLLMJSONStrict:
    def test_strict_object(self) -> None:
        assert _parse_llm_json('{"a": 1}') == {"a": 1}

    def test_strict_array(self) -> None:
        assert _parse_llm_json("[1, 2, 3]") == [1, 2, 3]

    def test_strict_nested(self) -> None:
        assert _parse_llm_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


class TestParseLLMJSONFenced:
    def test_fenced_with_json_tag(self) -> None:
        content = '```json\n{"a": 1}\n```'
        assert _parse_llm_json(content) == {"a": 1}

    def test_fenced_without_tag(self) -> None:
        content = '```\n{"a": 1}\n```'
        assert _parse_llm_json(content) == {"a": 1}

    def test_fenced_with_preamble(self) -> None:
        content = 'Here is the JSON:\n```json\n{"answer": 42}\n```\nThanks!'
        assert _parse_llm_json(content) == {"answer": 42}

    def test_fenced_array(self) -> None:
        content = "```json\n[1, 2, 3]\n```"
        assert _parse_llm_json(content) == [1, 2, 3]


class TestParseLLMJSONRawDecode:
    def test_extra_data_after_object(self) -> None:
        content = '{"a": 1}\nExtra commentary that breaks strict json.loads'
        assert _parse_llm_json(content) == {"a": 1}

    def test_two_concatenated_objects(self) -> None:
        content = '{"a": 1}{"b": 2}'
        assert _parse_llm_json(content) == {"a": 1}

    def test_leading_whitespace_then_object(self) -> None:
        content = '   \n{"a": 1}\ntrailing'
        assert _parse_llm_json(content) == {"a": 1}


class TestParseLLMJSONFailure:
    def test_all_strategies_fail(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_json("this is not json at all")

    def test_empty_string(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_llm_json("")
