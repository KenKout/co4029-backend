"""Lesson discussions feature.

A teacher who can manage a course posts a discussion **topic** (open
questions / prompt) on a lesson; enrolled students post **comments** to
discuss. See ``models.py`` for the two-table schema.

Models are imported here (mirroring the quizzes feature) so the SQLAlchemy
mapper registry discovers them even when only the package is imported.
"""

from abridgeai.features.discussions.models import (
    LessonDiscussionComment,
    LessonDiscussionTopic,
)

__all__ = [
    "LessonDiscussionComment",
    "LessonDiscussionTopic",
]
