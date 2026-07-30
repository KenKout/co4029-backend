"""Public re-exports for the quizzes-feature schema module (T5.2).

Four concerns split across sibling files:

* :mod:`.public`     — student-facing DTOs. Critical security
  invariant: ``QuizQuestionOptionPublic`` does NOT expose
  ``is_correct``. ``QuizPublic.status`` narrows to
  ``Literal["published"]``.
* :mod:`.authoring`  — teacher-facing DTOs that inherit from Public
  and widen with ``is_correct`` (per option), full status enum, audit
  + soft-delete + review-pipeline metadata.
* :mod:`.run`        — request / response shapes for generation +
  regeneration endpoints (``POST /generate``, ``GET /runs/{id}``).
* :mod:`.attempt`    — student attempt request / response shapes.
"""

from __future__ import annotations

from abridgeai.features.quizzes.schemas.attempt import (
    QuizAttemptIntegrityEvent,
    QuizAttemptProgressAnswer,
    QuizAttemptProgressRead,
    QuizAttemptRead,
    QuizAttemptReviewOption,
    QuizAttemptReviewQuestion,
    QuizAttemptReviewRead,
    QuizAttemptStart,
    QuizAttemptStatusLiteral,
    QuizAttemptSubmit,
    QuizAttemptSubmitAnswer,
    QuizAttemptTeacherRead,
    QuizAttemptTeacherReview,
)
from abridgeai.features.quizzes.schemas.authoring import (
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizQuestionAuthoring,
    QuizQuestionOptionAuthoring,
)
from abridgeai.features.quizzes.schemas.bank import (
    QuestionBankEntry,
    QuestionBankImportRequest,
    QuestionBankPage,
)
from abridgeai.features.quizzes.schemas.public import (
    QuestionTypeLiteral,
    QuizForTakingPublic,
    QuizPublic,
    QuizQuestionOptionPublic,
    QuizQuestionPublic,
)
from abridgeai.features.quizzes.schemas.feedback import (
    FeedbackBandIn,
    FeedbackBandRead,
    OverallFeedbackRead,
    QuizGradeRow,
)
from abridgeai.features.quizzes.schemas.reports import (
    ResponsesReportRead,
    ResponsesReportRow,
    StatisticsReportRead,
    StatisticsReportRow,
)
from abridgeai.features.quizzes.schemas.manual_grading import (
    ManualGradeIn,
    ManualGradeRead,
    NeedsGradingRow,
)
from abridgeai.features.quizzes.schemas.overrides import (
    QuizOverrideIn,
    QuizOverrideRead,
)
from abridgeai.features.quizzes.schemas.regrade import (
    RegradeItemRead,
    RegradeRunRead,
    RegradeScopeIn,
)
from abridgeai.features.quizzes.schemas.review_options import (
    ReviewOptions,
    ReviewWindowFlags,
)
from abridgeai.features.quizzes.schemas.results import (
    QuizOptionDistribution,
    QuizPerStudentRow,
    QuizQuestionBreakdown,
    QuizResultsRead,
    QuizResultsSummary,
    QuizScoreBucket,
)
from abridgeai.features.quizzes.schemas.run import (
    CoverageOptions,
    GenerationMode,
    GenerationRunStatus,
    QuestionRegenerationRequest,
    QuestionType,
    QuizGenerationProgress,
    QuizGenerationRequest,
    QuizGenerationRunRead,
    QuizGenerationStageEvent,
)

__all__ = [
    "CoverageOptions",
    "GenerationMode",
    "GenerationRunStatus",
    "QuestionBankEntry",
    "QuestionBankImportRequest",
    "QuestionBankPage",
    "QuestionRegenerationRequest",
    "QuestionType",
    "QuestionTypeLiteral",
    "QuizAttemptIntegrityEvent",
    "QuizAttemptProgressAnswer",
    "QuizAttemptProgressRead",
    "QuizAttemptRead",
    "QuizAttemptReviewOption",
    "QuizAttemptReviewQuestion",
    "QuizAttemptReviewRead",
    "QuizAttemptStart",
    "QuizAttemptStatusLiteral",
    "QuizAttemptSubmit",
    "QuizAttemptSubmitAnswer",
    "QuizAttemptTeacherRead",
    "QuizAttemptTeacherReview",
    "QuizAuthoring",
    "QuizForAuthoringPublic",
    "QuizForTakingPublic",
    "QuizGenerationProgress",
    "QuizGenerationRequest",
    "QuizGenerationRunRead",
    "QuizGenerationStageEvent",
    "QuizOptionDistribution",
    "QuizPerStudentRow",
    "QuizPublic",
    "QuizQuestionAuthoring",
    "QuizQuestionBreakdown",
    "QuizQuestionOptionAuthoring",
    "QuizQuestionOptionPublic",
    "QuizQuestionPublic",
    "QuizResultsRead",
    "QuizResultsSummary",
    "QuizScoreBucket",
    "FeedbackBandIn",
    "FeedbackBandRead",
    "ManualGradeIn",
    "ManualGradeRead",
    "NeedsGradingRow",
    "OverallFeedbackRead",
    "QuizGradeRow",
    "QuizOverrideIn",
    "QuizOverrideRead",
    "RegradeItemRead",
    "RegradeRunRead",
    "RegradeScopeIn",
    "ResponsesReportRead",
    "ResponsesReportRow",
    "StatisticsReportRead",
    "StatisticsReportRow",
    "ReviewOptions",
    "ReviewWindowFlags",
]
