"""ARQ worker package.

``set_worker_actor`` is re-exported here as the binder every task body
calls first per the Phase 0.8 convention. ``WorkerSettings`` lives in
``abridgeai.workers.arq_app`` and is intentionally NOT re-exported from
this package init — importing it here would create a cycle with feature
workers that need ``set_worker_actor`` (they would trigger this package
init, which would trigger ``arq_app``, which imports the feature
``JOBS`` lists). Run ARQ with the explicit module path:

    arq abridgeai.workers.arq_app.WorkerSettings
"""

from abridgeai.workers.actor import set_worker_actor

__all__ = ["set_worker_actor"]
