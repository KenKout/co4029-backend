"""Materials feature — typed cross-feature read surface (T22).

Sibling features import only from :mod:`abridgeai.features.materials.api.public`.
Internal modules (``models``, ``queries``, ``services``, ``routers``) are
not re-exported here; the import-linter "Features are independent"
contract enforces that boundary.
"""

from abridgeai.features.materials.api import public

__all__ = ["public"]
