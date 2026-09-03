"""Platform policy documents as a versioned, admin-editable entity.

Models are imported here (mirroring the quizzes/discussions features) so the
SQLAlchemy mapper registry discovers them when only the package is imported.
"""

from abridgeai.features.policies.models import (
    Policy,
    PolicyAudienceRole,
    PolicyVersion,
)

__all__ = [
    "Policy",
    "PolicyAudienceRole",
    "PolicyVersion",
]
