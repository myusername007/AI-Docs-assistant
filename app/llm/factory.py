from functools import lru_cache

from app.core.config import settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient
from app.llm.gemini_client import GeminiClient
from app.llm.openai_client import OpenAIClient

_REGISTRY: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def build_client(provider: str) -> LLMClient:
    try:
        return _REGISTRY[provider]()
    except KeyError:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            f"Available: {', '.join(_REGISTRY)}"
        ) from None


@lru_cache
def get_llm_client() -> LLMClient:
    """Cached default client based on settings.llm_provider."""
    return build_client(settings.llm_provider)