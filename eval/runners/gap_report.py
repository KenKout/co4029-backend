"""Gap report runner. Wires `gap_report` capability to the
`abridgeai.features.progress.ai` reporting pipeline. T8.2 ships the stub.
"""

from __future__ import annotations

from eval.runners.base import CapabilityRunner


class GapReportRunner(CapabilityRunner):
    capability = "gap_report"
