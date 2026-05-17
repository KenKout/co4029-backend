"""Public API of the LLM integration package.

External callers should import from ``abridgeai.ai.llm`` directly:

    from abridgeai.ai.llm import LLMGateway, LLMRole, EmbeddingClient

Internal modules can import from submodules to avoid circular references.
"""

from abridgeai.ai.llm.embeddings import EmbeddingClient
from abridgeai.ai.llm.errors import ConfigError, ProviderError, ResponseFormatError
from abridgeai.ai.llm.gateway import LLMGateway, LLMResult
from abridgeai.ai.llm.roles import LLMRole, ModelBinding

__all__ = [
    "ConfigError",
    "EmbeddingClient",
    "LLMGateway",
    "LLMResult",
    "LLMRole",
    "ModelBinding",
    "ProviderError",
    "ResponseFormatError",
]
