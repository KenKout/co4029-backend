"""Add interview_configs.tts_voice (teacher-selectable Deepgram Aura voice)

Revision ID: 0043_interview_tts_voice
Revises: 0042_gen_type_interview_eval
Create Date: 2026-07-24 00:00:00.000000

Teachers can pick the English TTS voice for an interview (a Deepgram Aura-2
model id, one voice per model — e.g. 'aura-2-thalia-en'). NULL means "use the
deployment default" (settings.deepgram_tts_model_en), so existing configs are
unaffected. Only English narration/voice honours this; Vietnamese has no server
TTS. Value is validated against a curated allow-list in the authoring schema,
so no DB CHECK is added (keeps the voice catalog editable in code).

Additive, nullable column — fully reversible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0043_interview_tts_voice"
down_revision = "0042_gen_type_interview_eval"
branch_labels = None
depends_on = None

_TABLE = "interview_configs"
_COLUMN = "tts_voice"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(length=40), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
