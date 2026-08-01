import time

from openai import AsyncOpenAI

from app.core.config import settings
from app.llm.base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    provider = "openai"
    default_model = "gpt-5.6-luna"

    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model = model or self.default_model

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            provider=self.provider,
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )