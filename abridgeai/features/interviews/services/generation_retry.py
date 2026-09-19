"""Transient-failure classification for the interview generation pipeline.

P2 (generation retry): a transient provider/transport error must not
terminally fail a generation run on its first attempt. The pipeline uses
:func:`is_transient_generation_failure` to decide whether a run goes back to
``pending`` (retry budget remaining) or ``failed`` (terminal); the ARQ worker
uses the same predicate to decide whether to raise ``arq.worker.Retry``.
"""

from __future__ import annotations

from abridgeai.ai.llm.errors import ConfigError, ProviderError

_RETRYABLE_PROVIDER_MARKERS = (
    # httpx transport failures ("HTTP error calling ...: ...")
    "HTTP error",
    # gateway 5xx
    "HTTP 5",
    # rate limited after in-client retries
    "HTTP 429",
)


def is_transient_generation_failure(exc: BaseException) -> bool:
    """True for failures worth another attempt.

    Transient: transport-level httpx errors, gateway 5xx, 429s (after the
    client's own bounded backoff is spent), and plain socket timeouts.
    Permanent: validation/not-found/config errors, unparseable model
    responses (:class:`ResponseFormatError`), and non-5xx provider
    rejections (bad request, auth) — retrying those cannot succeed.
    """
    if isinstance(exc, (ConfigError,)):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, ProviderError):
        message = str(exc)
        return any(marker in message for marker in _RETRYABLE_PROVIDER_MARKERS)
    return False


__all__ = ["is_transient_generation_failure"]
