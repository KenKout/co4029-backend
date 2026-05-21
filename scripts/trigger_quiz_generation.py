"""Trigger quiz generation directly via the service layer (no HTTP).

Bypasses OAuth so we can smoke-test the full Phase 1-4 contextual-RAG
stack on the live backend. Mirrors what
``services.authoring.create_quiz_with_generation`` does:

  1. Build the FR-5 ``generation_config`` dict.
  2. Create + persist a :class:`GenerationRun` row.
  3. Enqueue ``run_quiz_generation_task(actor_id, run_id)`` on ARQ.
  4. Print the run id; tail the worker log to follow progress.
"""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import UUID, uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arq.connections import ArqRedis, RedisSettings, create_pool  # type: ignore[import-untyped]
from sqlalchemy import select

# Importing each feature's models module wires up cross-table FK metadata.
# Without this, accessing GenerationRun.requested_by raises NoReferencedTableError.
from abridgeai.ai import models as _ai_models  # noqa: F401
from abridgeai.ai.models import GenerationRun
from abridgeai.core.db import get_sessionmaker
from abridgeai.features.access_control import models as _access_models  # noqa: F401
from abridgeai.features.courses import models as _course_models  # noqa: F401
from abridgeai.features.materials import models as _material_models  # noqa: F401
from abridgeai.features.quizzes.models import Quiz, QuizQuestion
from abridgeai.features.identity import models as _user_models  # noqa: F401
from abridgeai.features.interviews import models as _interview_models  # noqa: F401
from abridgeai.features.notifications import models as _notif_models  # noqa: F401
from abridgeai.features.progress import models as _progress_models  # noqa: F401
from abridgeai.features.spaced_repetition import models as _sr_models  # noqa: F401

QUIZ_ID = UUID("5ffec213-a639-4e73-85cb-65f1255f85f0")
LESSON_ID = UUID("46741c12-ff9d-48ac-9804-24753f6386eb")


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL", "")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    Session = get_sessionmaker()

    async with Session() as db:
        quiz = (
            await db.execute(select(Quiz).where(Quiz.id == QUIZ_ID))
        ).scalar_one()
        print(f"Found quiz: {quiz.title!r}, module={quiz.module_id}")

        # Wipe existing questions so we get a clean regeneration.
        from sqlalchemy import delete as sa_delete

        from abridgeai.features.quizzes.models import QuizQuestion

        await db.execute(sa_delete(QuizQuestion).where(QuizQuestion.quiz_id == QUIZ_ID))

        config = {
            "quiz_id": str(QUIZ_ID),
            "question_count": 8,
            "question_types": [
                "multiple_choice",
                "true_false",
                "short_answer",
                "fill_blank",
            ],
            "difficulty": "medium",
            "bloom_distribution": {"remember": 0.4, "understand": 0.4, "apply": 0.2},
            "include_prerequisites": False,
            "model_preference": None,
            "source_lesson_ids": [str(LESSON_ID)],
            "generation_mode": "manual",
            "focus_topics": [],
            "avoid_topics": [],
            "extra_instructions": "",
            "append": False,
            "coverage_options": {
                "coverage_threshold": 0.7,
                "slides_per_section": 4,
                "section_grouping": "auto",
            },
        }
        # actor_id: original quiz creator — required for ai_model_calls audit FK.
        actor_id = quiz.created_by or uuid4()
        run = GenerationRun(
            generation_type="quiz",
            source_scope_kind="module",
            course_id=quiz.course_id,
            module_id=quiz.module_id,
            requested_by=actor_id,
            status="pending",
            config_json=config,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)
        # Link the run to the quiz so the worker writes back to it.
        quiz.generation_run_id = run.id
        await db.commit()

        print(f"Created GenerationRun {run.id} (status={run.status})")

    redis: ArqRedis = await create_pool(RedisSettings.from_dsn(redis_url))
    job = await redis.enqueue_job(
        "run_quiz_generation_task",
        actor_id,
        run.id,
    )
    if job is not None:
        print(f"Enqueued arq job: {job.job_id}")
    else:
        print("WARN — enqueue returned None (job already queued?)")
    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())
