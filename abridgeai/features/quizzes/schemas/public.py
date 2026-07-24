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

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

QuestionTypeLiteral = Literal[
    "multiple_choice",
    "true_false",
    "short_answer",
    "fill_blank",
    "code",
]


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
    options: list[QuizQuestionOptionPublic] = []

    # Course learning outcome this question assesses (§LO-3). The FK is
    # stored on the row; ``outcome_position`` is NOT — it's filled by the
    # projection layer (batch lookup) so the client can render the derived
    # ``(L.O.{outcome_position})`` prefix without an ORM relationship
    # crossing the quizzes→courses feature boundary. Both NULL = no outcome.
    learning_outcome_id: UUID | None = None
    outcome_position: int | None = None


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
