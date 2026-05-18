"""Public API surface for the access_control feature.

Cross-feature consumers MUST import from
:mod:`abridgeai.features.access_control.api.public` rather than reaching
into ``models``/``policies``/``queries`` directly. The independence
contract in ``pyproject.toml`` (``Features are independent``) is enforced
against ``abridgeai.features.*`` packages; this ``api`` subpackage is the
blessed import surface.
"""
