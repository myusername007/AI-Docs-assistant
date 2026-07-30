import time

from anthropic import AsyncAnthropic

from app.core.config import settings
from app.llm.base import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    provider = "anthropic"
    default_model = "claude-sonnet-5"

    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model = model or self.default_model

        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system

        started = time.perf_counter()
        response = await self._client.messages.create(**params)
        latency_ms = (time.perf_counter() - started) * 1000

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return LLMResponse(
            text=text,
            provider=self.provider,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )