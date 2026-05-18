"""Public API surface for the enrollments feature.

Cross-feature consumers MUST import from
:mod:`abridgeai.features.enrollments.api.public` rather than reaching
into ``models``/``queries``/``services`` directly. The independence
contract in ``pyproject.toml`` (``Features are independent``) is enforced
against ``abridgeai.features.*`` packages; this ``api`` subpackage is
the blessed import surface.
"""
