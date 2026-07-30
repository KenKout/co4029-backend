"""Transcript shaping for the post-hoc quality metrics.

Pure helpers (no DB, no LLM) that turn a flat list of session messages into the
two shapes the judges need:

* :func:`build_qa_pairs` — every interviewer utterance paired with the student
  answer that followed it, in order. Used by the *leading question* judge, which
  needs the answer to tell "the student volunteered this" from "the interviewer
  fed it to them".
* :func:`followup_pairs` — only those interviewer turns that are follow-ups on
  the SAME question (a probe / clarification / challenge) rather than a move to a
  new question, paired with the student answer they are responding to. Used by
  the *contingency* judge.

Distinguishing a follow-up from a new question matters: judging "did this build
on the previous answer?" against a deliberate topic change would penalise
correct behaviour. We use ``session_question_id`` — a follow-up keeps the same
one — rather than trying to infer intent from the text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptTurn:
    """One transcript message, decoupled from the ORM row."""

    message_id: str
    role: str  # "ai" | "user" | "system"
    text: str
    session_question_id: str | None = None
    sequence_no: int = 0


@dataclass(frozen=True)
class QAPair:
    """An interviewer utterance and the student answer that followed it."""

    interviewer_message_id: str
    interviewer_text: str
    student_text: str
    session_question_id: str | None = None
    # The student answer immediately BEFORE the interviewer utterance, when the
    # utterance was a follow-up. This is what a contingent probe must build on.
    preceding_student_text: str | None = None

    @property
    def is_followup(self) -> bool:
        return self.preceding_student_text is not None


def _clean(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Drop system messages and blanks; keep the caller's ordering."""
    return [t for t in turns if t.role in {"ai", "user"} and (t.text or "").strip()]


def build_qa_pairs(turns: list[TranscriptTurn]) -> list[QAPair]:
    """Pair each interviewer utterance with the student answer that followed.

    An interviewer turn with no student answer after it (e.g. the closing
    remark, or an abandoned session) is skipped — there is nothing to judge
    contamination against. Consecutive interviewer turns are handled by pairing
    each with the next student answer that appears.
    """
    clean = _clean(turns)
    pairs: list[QAPair] = []
    for idx, turn in enumerate(clean):
        if turn.role != "ai":
            continue
        student_after = next((t for t in clean[idx + 1 :] if t.role == "user"), None)
        if student_after is None:
            continue
        student_before = next(
            (t for t in reversed(clean[:idx]) if t.role == "user"),
            None,
        )
        # Same session_question_id as the answer before it → this utterance is a
        # follow-up on that answer rather than a new question.
        is_followup = (
            student_before is not None
            and turn.session_question_id is not None
            and turn.session_question_id == student_before.session_question_id
        )
        pairs.append(
            QAPair(
                interviewer_message_id=turn.message_id,
                interviewer_text=turn.text.strip(),
                student_text=student_after.text.strip(),
                session_question_id=turn.session_question_id,
                preceding_student_text=(
                    student_before.text.strip() if is_followup and student_before else None
                ),
            )
        )
    return pairs


def followup_pairs(turns: list[TranscriptTurn]) -> list[QAPair]:
    """Only the follow-up utterances, for the contingency judge.

    Empty result is itself a finding: an interviewer that never follows up is the
    scripted-checklist failure mode, and the caller reports it as such rather
    than as "no data".
    """
    return [p for p in build_qa_pairs(turns) if p.is_followup]


__all__ = ["QAPair", "TranscriptTurn", "build_qa_pairs", "followup_pairs"]
