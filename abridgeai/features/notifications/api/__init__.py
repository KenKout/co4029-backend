"""Public API surface for the notifications feature.

Cross-feature producers (SR remediation, SR scan-due-cards, future:
quizzes/interviews/enrollments completion events) MUST import the
dispatch entrypoint from
:mod:`abridgeai.features.notifications.api.public` rather than reaching
into ``services.dispatch`` directly. The independence contract in
``pyproject.toml`` (``Features are independent``) is enforced against
``abridgeai.features.*`` packages; this ``api`` subpackage is the
blessed import surface.
"""
