"""Public API surface for the interviews feature.

Cross-feature consumers (progress dashboards, admin reports) MUST import
from :mod:`abridgeai.features.interviews.api.public` rather than reaching
into ``models``/``queries``/``services`` directly. The independence
contract in ``pyproject.toml`` (``Features are independent``) is enforced
against ``abridgeai.features.*`` packages; this ``api`` subpackage is
the blessed import surface.
"""
