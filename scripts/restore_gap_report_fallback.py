"""One-off remediation: restore gap_report 4a254f12 (session 677bf2e6) from the
stored LLM response so the student sees the real AI feedback again.

The gap-report boundary previously fell back to "could not be displayed
safely" (false-positive internal_prompt token overlap) and blanked
strengths/weaknesses/study_plan. The original LLM content survives in
ai_model_calls.response_payload. This restores exactly that content.
"""

import asyncio
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from abridgeai.core.config import get_settings

SESSION_ID = "677bf2e6-4d97-4be2-89c1-ff11f20e3a30"
GENERATION_RUN_ID = "205112ec-365a-40ed-a0d9-f16423330cca"


async def main() -> None:
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT response_payload FROM ai_model_calls "
                    "WHERE stage_name='gap_report' AND generation_run_id=:rid"
                ),
                {"rid": GENERATION_RUN_ID},
            )
        ).first()
        if row is None:
            print("NO gap_report model call row found")  # noqa: T201
            return
        payload = row[0]
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            content = json.loads(content)
        print("LLM content keys:", sorted(content.keys()))  # noqa: T201

        report_json = {
            "strengths": content["strengths"],
            "weaknesses": content["weaknesses"],
            "study_plan": content["study_plan"],
        }
        await conn.execute(
            text(
                "UPDATE gap_reports SET student_summary=:summary, "
                "report_json = report_json || :parts "
                "WHERE source_interview_session_id=:session_id"
            ),
            {
                "summary": content["student_summary"],
                "parts": json.dumps(report_json),
                "session_id": SESSION_ID,
            },
        )
        print("restored student_summary + strengths/weaknesses/study_plan")  # noqa: T201

        check = (
            await conn.execute(
                text(
                    "SELECT LEFT(student_summary, 80) AS s, "
                    "jsonb_array_length(report_json->'study_plan') AS plan, "
                    "jsonb_array_length(report_json->'strengths') AS strengths "
                    "FROM gap_reports WHERE source_interview_session_id=:sid"
                ),
                {"sid": SESSION_ID},
            )
        ).first()
        print("verification:", check)  # noqa: T201
    await engine.dispose()


asyncio.run(main())
