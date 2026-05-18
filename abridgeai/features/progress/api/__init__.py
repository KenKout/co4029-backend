"""Public API surface for the progress feature.

Cross-feature consumers (admin dashboards, monitoring) MUST import from
:mod:`abridgeai.features.progress.api.public` rather than reaching into
``models``/``queries``/``services`` directly. The independence contract
in ``pyproject.toml`` (``Features are independent``) is enforced against
``abridgeai.features.*`` packages; this ``api`` subpackage is the
blessed import surface.
"""
