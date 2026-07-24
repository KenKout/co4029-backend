"""Moodle XML import parser + export serializer (Phase 11).

Pure functions — no DB. Uses ``lxml.etree``. ``parse_moodle_xml`` turns a Moodle
question-bank XML file into a :class:`ParseResult`; ``serialize_moodle_xml``
turns parsed questions back into Moodle XML. Unsupported ``<question type>``
values are collected as warnings; malformed XML raises :class:`XmlParseError`.
"""

from __future__ import annotations

from lxml import etree

from abridgeai.features.quizzes.services.formats._types import (
    ParsedOption,
    ParsedQuestion,
    ParseResult,
)

_TYPE_MAP = {
    "multichoice": "multiple_choice",
    "truefalse": "true_false",
    "shortanswer": "short_answer",
    "essay": "code",
    "numerical": "numerical",
    "matching": "matching",
}
_REVERSE_TYPE_MAP = {
    "multiple_choice": "multichoice",
    "true_false": "truefalse",
    "short_answer": "shortanswer",
    "code": "essay",
    "numerical": "numerical",
    "matching": "matching",
    "ordering": "ordering",
}


class XmlParseError(ValueError):
    """Raised when the XML is not well-formed."""


def _text(el: object) -> str:
    if el is None:
        return ""
    node = el.find("text")  # type: ignore[attr-defined]
    return (node.text or "").strip() if node is not None and node.text else ""


def _frac(answer_el: object) -> float:
    try:
        return float(answer_el.get("fraction", "0"))  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return 0.0


def parse_moodle_xml(xml: str) -> ParseResult:
    try:
        root = etree.fromstring(xml.encode("utf-8"))
    except etree.XMLSyntaxError as exc:
        raise XmlParseError(str(exc)) from exc

    questions: list[ParsedQuestion] = []
    warnings: list[str] = []
    for i, qel in enumerate(root.findall("question"), start=1):
        mtype = qel.get("type", "")
        if mtype == "category":
            continue
        atype = _TYPE_MAP.get(mtype)
        if atype is None:
            warnings.append(f"Q{i}: unsupported Moodle type '{mtype}' skipped")
            continue
        prompt = _text(qel.find("questiontext"))
        gfb = _text(qel.find("generalfeedback")) or None
        answers = qel.findall("answer")
        if atype == "short_answer":
            correct = next((_text(a) for a in answers if _frac(a) >= 100.0), None)
            questions.append(
                ParsedQuestion(
                    question_type=atype,
                    prompt_text=prompt,
                    correct_answer=correct,
                    explanation=gfb,
                )
            )
            continue
        if atype == "code":
            questions.append(
                ParsedQuestion(question_type=atype, prompt_text=prompt, explanation=gfb)
            )
            continue
        opts = [ParsedOption(text=_text(a), is_correct=_frac(a) >= 100.0) for a in answers]
        questions.append(
            ParsedQuestion(
                question_type=atype, prompt_text=prompt, options=opts, explanation=gfb
            )
        )
    return ParseResult(questions=questions, warnings=warnings)


def serialize_moodle_xml(questions: list[ParsedQuestion]) -> str:
    """Serialize parsed questions to a Moodle question-bank XML string."""
    root = etree.Element("quiz")
    for q in questions:
        mtype = _REVERSE_TYPE_MAP.get(q.question_type, "multichoice")
        qel = etree.SubElement(root, "question", type=mtype)
        name = etree.SubElement(qel, "name")
        etree.SubElement(name, "text").text = q.prompt_text[:60] or "Question"
        qtext = etree.SubElement(qel, "questiontext", format="html")
        ct = etree.SubElement(qtext, "text")
        ct.text = etree.CDATA(q.prompt_text)
        if q.explanation:
            gfb = etree.SubElement(qel, "generalfeedback", format="html")
            etree.SubElement(gfb, "text").text = q.explanation
        if q.question_type == "short_answer" and q.correct_answer:
            ans = etree.SubElement(qel, "answer", fraction="100")
            etree.SubElement(ans, "text").text = q.correct_answer
        else:
            for o in q.options:
                ans = etree.SubElement(
                    qel, "answer", fraction=("100" if o.is_correct else "0")
                )
                etree.SubElement(ans, "text").text = o.text
    return etree.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
    ).decode("utf-8")


__all__ = ["XmlParseError", "parse_moodle_xml", "serialize_moodle_xml"]
