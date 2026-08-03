"""Student-facing (public) DTOs for the quizzes feature (T5.2).

These schemas back the learner slice of the quizzes router (T5.10) and
represent the narrowest projection of the quiz aggregate — the security
invariant for this module is that **answer correctness must NEVER leak
to students**. Concretely:

* :class:`QuizQuestionOptionPublic` does **NOT** expose ``is_correct``.
  Authoring callers want it (teachers grade and review); learners must
  only see the option text + key + position.
* :class:`QuizPublic` narrows ``status`` to ``Literal["published"]`` —
  draft / archived quizzes never surface here.
* Internal-only columns (``internal_notes``, ``draft_count``,
  ``review_status``, ``original_generated_payload``, audit / soft-delete
  metadata, generation-run linkage) are gated behind :mod:`.authoring`.

Reconciliation directives (HIGHER PRECEDENCE — see plan §266-585):

* §A13 — baseline DDL is ground truth. Field names mirror T5.1 ORM
  columns: ``prompt_text`` (not ``prompt``), ``option_text`` (not
  ``content``), ``passing_score_percent`` (not ``passing_score``).
  ``shuffle_questions`` / ``shuffle_options`` are exposed as authoring
  toggles only — the public DTO carries the resolved order.
* §C1 — ``QuizQuestion.question_type`` CHECK enforces
  ``{multiple_choice, true_false, short_answer, fill_blank, code}``
  after migration 0007's ``mcq → multiple_choice`` data migration. The
  ``Literal`` here mirrors that CHECK byte-for-byte.
* §C7 — ``QuizQuestion.hint_text`` is unbounded ``Text`` (not
  ``String(500)``); a learner may request the hint via the take-quiz
  flow, so ``hint_text`` IS exposed publicly (gated by quiz config
  ``show_hints`` at the service layer, not at the schema layer).

Field drops vs plan body §5439-5442 (per T5.1 ORM ground truth, §A13)
--------------------------------------------------------------------
The plan body suggests several fields that do NOT exist in the T5.1
ORM (which mirrors baseline DDL):

* :class:`QuizPublic`: ``passing_score`` does not exist — column is
  ``passing_score_percent`` (NUMERIC(5,2)). ``is_randomized`` does not
  exist — split into ``shuffle_questions`` / ``shuffle_options`` (and
  both are authoring-only toggles, hidden from the public DTO).

If a future migration adds new columns, this module is the only place
that needs updating; authoring schemas continue to inherit + widen
automatically.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

QuestionTypeLiteral = Literal[
    "multiple_choice",
    "true_false",
    "short_answer",
    "fill_blank",
    "code",
    # Phase 7: expanded question types.
    "numerical",
    "matching",
    "ordering",
]


def _field_allows_none(field: Any) -> bool:  # noqa: ANN401  -- pydantic FieldInfo
    """True when a model field's annotation permits ``None``.

    Used by the no-leak merge below to tell ``hint_text: str | None`` (where a
    source None is meaningful) apart from ``prompt_format: str = "plain"`` (where
    a source None is just an unmaterialized server_default and must fall back to
    the schema default).
    """
    annotation = field.annotation
    if annotation is None or annotation is type(None):
        return True
    return type(None) in get_args(annotation)


class _ORMModel(BaseModel):
    """Shared base — Pydantic v2 ORM-mode equivalent (`from_attributes=True`)."""

    model_config = ConfigDict(from_attributes=True)


class QuizQuestionOptionPublic(_ORMModel):
    """Student-facing projection of one ``QuizQuestionOption`` row.

    SECURITY INVARIANT: ``is_correct`` MUST NOT appear in this schema.
    Adding it would leak answer correctness to learners taking the
    quiz. The authoring counterpart :class:`QuizQuestionOptionAuthoring`
    re-introduces it for teacher screens.
    """

    id: UUID
    option_key: str
    option_text: str
    position: int
    # Phase 3: render discriminator for option_text (plain | markdown | html).
    option_format: str = "plain"


class QuizQuestionPublic(_ORMModel):
    """Student-facing projection of one ``QuizQuestion`` row.

    Carries only what the learner needs to render and answer the
    question: identity, prompt, type discriminator, options, and the
    optional hint (gated by ``Quiz.show_hints`` at the service layer).

    Authoring-only fields hidden here:

    * ``review_status``, ``original_generated_payload``,
      ``source_refs``, ``reviewed_by`` / ``reviewed_at``,
      ``published_at`` — generation / review pipeline metadata.
    * ``difficulty``, ``bloom_level``, ``expected_response_time_ms``,
      ``expected_ef_ceiling`` — analytics + SR scheduler inputs.
    * ``explanation`` — only revealed AFTER the attempt is submitted
      (T5.11 review payload composes the explanation separately).
    * Audit + soft-delete columns.
    """

    id: UUID
    quiz_id: UUID
    position: int
    question_type: QuestionTypeLiteral
    prompt_text: str
    hint_text: str | None = None
    # Phase 3: render discriminators (plain | markdown | html).
    prompt_format: str = "plain"
    hint_format: str = "plain"
    # Phase 7: multi-select discriminator. True (default) = single-answer MCQ
    # (radio); False = multi-select (checkboxes). Safe to expose — it reveals
    # the input shape, not which options are correct.
    single_answer: bool = True
    options: list[QuizQuestionOptionPublic] = []

    # Course learning outcome this question assesses (§LO-3). The FK is
    # stored on the row; ``outcome_position`` is NOT — it's filled by the
    # projection layer (batch lookup) so the client can render the derived
    # ``(L.O.{outcome_position})`` prefix without an ORM relationship
    # crossing the quizzes→courses feature boundary. Both NULL = no outcome.
    learning_outcome_id: UUID | None = None
    outcome_position: int | None = None
    # Dotted display code of the assessed outcome (e.g. "1.2.1" → rendered
    # "L.O.1.2.1"). Projection-only, filled alongside outcome_position from a
    # batch lookup. Supersedes the flat outcome_position for display now that
    # outcomes are hierarchical; outcome_position is kept for back-compat
    # (top-level rows: code == str(position)).
    outcome_code: str | None = None

    # Phase 7: multi-select discriminator. True (default) = single-answer MCQ
    # (radio); False = multi-select (checkboxes). Safe — reveals the input
    # shape, not which options are correct.
    single_answer: bool = True

    # Phase 7 no-leak projections for matching / ordering. The raw answer keys
    # (``match_pairs`` = [{left, right}], ``ordering_sequence`` = correct order)
    # MUST NOT reach a learner. Instead we derive:
    #   * ``match_prompts``  — the left column, in stem order (safe to show).
    #   * ``match_choices``  — the right values, DETERMINISTICALLY SHUFFLED so
    #     the pairing is not implied by position.
    #   * ``ordering_items`` — the items, DETERMINISTICALLY SHUFFLED so the
    #     correct sequence is never revealed.
    # The shuffle is seeded by the question id → stable across polls (won't
    # reshuffle mid-attempt) but not in answer order. Correctness is graded
    # server-side against the hidden keys.
    match_prompts: list[str] = []
    match_choices: list[str] = []
    ordering_items: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _derive_no_leak_type_fields(cls, data: Any) -> Any:
        """Build the shuffled matching/ordering projections from the ORM row.

        Runs before field validation. Reads the answer-bearing columns off the
        source object (ORM instance or dict) and injects only the safe derived
        lists — the raw ``match_pairs`` / ``ordering_sequence`` are never copied
        onto the schema, so they cannot serialize to the wire.
        """

        def _get(key: str) -> Any:
            if isinstance(data, dict):
                return data.get(key)
            return getattr(data, key, None)

        qid = _get("id")
        seed = 0
        if qid is not None:
            try:
                seed = int(uuid.UUID(str(qid)).hex, 16)
            except (ValueError, AttributeError, TypeError):
                seed = hash(str(qid))

        derived: dict[str, list[str]] = {}

        pairs = _get("match_pairs")
        if isinstance(pairs, list) and pairs:
            prompts = [str(p.get("left", "")) for p in pairs if isinstance(p, dict)]
            answer_aligned = [str(p.get("right", "")) for p in pairs if isinstance(p, dict)]
            # Distractors are extra right-side values with no left partner. They
            # go into the choice pool the learner picks from — never into the
            # prompts — so the last prompt isn't a forced pick and elimination is
            # harder. Dedupe against the real answers (a distractor equal to a
            # correct value would make that value un-wrong, defeating the point).
            raw_distractors = _get("match_distractors")
            answer_set = {a.strip().lower() for a in answer_aligned}
            distractors = (
                [
                    d
                    for d in (str(x).strip() for x in raw_distractors)
                    if d and d.lower() not in answer_set
                ]
                if isinstance(raw_distractors, list)
                else []
            )
            choices = [*answer_aligned, *distractors]
            rng = random.Random(seed ^ 0x9E3779B9)
            # Reshuffle until the choice order no longer positionally aligns with
            # the prompts (which would imply the pairing on small lists). With
            # distractors present the lengths differ, so alignment is impossible
            # and one shuffle suffices — the guard still holds for the 1:1 case.
            for _ in range(8):
                rng.shuffle(choices)
                if choices[: len(answer_aligned)] != answer_aligned or len(choices) <= 1:
                    break
            derived["match_prompts"] = prompts
            derived["match_choices"] = choices

        sequence = _get("ordering_sequence")
        if isinstance(sequence, list) and sequence:
            items = [str(s) for s in sequence]
            rng = random.Random(seed ^ 0x7F4A7C15)
            # Guard against the shuffle accidentally reproducing the answer
            # order on tiny lists — reshuffle until it differs (bounded).
            for _ in range(8):
                rng.shuffle(items)
                if items != [str(s) for s in sequence] or len(items) <= 1:
                    break
            derived["ordering_items"] = items

        if not derived:
            return data

        # Merge derived fields with the source. For an ORM object we build a
        # dict of the declared fields so Pydantic still reads the rest. The
        # three derived list fields are only injected when computed; otherwise
        # we omit them so their schema default (``[]``) applies (a raw getattr
        # would return None on the ORM row and break list validation).
        _DERIVED_FIELDS = {"match_prompts", "match_choices", "ordering_items"}
        if isinstance(data, dict):
            return {**data, **derived}
        merged: dict[str, Any] = {}
        _MISSING = object()
        for name in cls.model_fields:
            if name in derived:
                merged[name] = derived[name]
                continue
            if name in _DERIVED_FIELDS:
                continue  # not computed → let the field default ([]) stand
            # Only copy attributes the source actually HAS. ``QuizQuestion`` has
            # no ``options`` ORM relationship (callers attach it manually), so a
            # blind ``getattr(..., None)`` would inject None into a required
            # list field and fail validation for any question loaded without it.
            value = getattr(data, name, _MISSING)
            if value is _MISSING:
                continue
            # Don't let a None from the source override a non-nullable field that
            # has a schema default. Columns backed by a server_default (e.g.
            # ``prompt_format``, ``single_answer``) read as None on a row that
            # hasn't been flushed/refreshed yet; copying that None would fail
            # validation even though the schema default is perfectly valid.
            if value is None:
                field = cls.model_fields[name]
                if not field.is_required() and not _field_allows_none(field):
                    continue
            merged[name] = value
        return merged


class QuizPublic(_ORMModel):
    """Student-facing quiz summary.

    ``status`` narrows to ``Literal["published"]`` — the public catalog
    only ever surfaces published quizzes. The authoring counterpart
    :class:`QuizAuthoring` widens to the full enum so teachers can see
    drafts.

    Authoring-only fields hidden here:

    * ``course_id`` / ``module_id`` — exposed via the parent course
      payload, not duplicated onto the quiz public DTO.
    * ``shuffle_questions`` / ``shuffle_options`` — server applies the
      shuffle when composing :class:`QuizForTakingPublic`; clients do
      not need the toggles.
    * SR-config knobs (``initial_ef``, ``min_ef_for_unlock``,
      ``coverage_threshold``, ``reminders_enabled``,
      ``generation_instructions``, ``generation_run_id``).
    * ``published_at`` and audit / soft-delete columns.
    """

    id: UUID
    title: str
    description: str | None = None
    status: Literal["published"]
    passing_score_percent: Decimal
    time_limit_seconds: int | None = None
    max_attempts: int | None = None
    allow_retakes: bool = True
    cooldown_hours: int | None = None
    show_hints: bool = True
    # Scheduling window (migration 0032). NULL = no restriction.
    available_from: datetime | None = None
    available_until: datetime | None = None
    due_at: datetime | None = None
    # SAFE expectation-setting signal: how many approved questions the student
    # will face. Exposes only the COUNT, never the question text / options /
    # is_correct flags — those remain in QuizQuestionPublic and are served
    # one-at-a-time via the taking payload. Mirrors the interview
    # ``outcome_count`` pattern. Defaults to 0 so a bare model_validate (without
    # the count wired) still validates.
    question_count: int = 0


class QuizForTakingPublic(_ORMModel):
    """Composed take-quiz payload — quiz + ordered question list.

    The service layer (T5.10) is responsible for honoring
    ``Quiz.shuffle_questions`` / ``shuffle_options`` when building the
    list. This DTO declares only the wire shape; ordering is the
    caller's concern.
    """

    quiz: QuizPublic
    questions: list[QuizQuestionPublic] = []


__all__ = [
    "QuestionTypeLiteral",
    "QuizForTakingPublic",
    "QuizPublic",
    "QuizQuestionOptionPublic",
    "QuizQuestionPublic",
]
