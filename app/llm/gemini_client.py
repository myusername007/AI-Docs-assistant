import time

from google import genai
from google.genai import types

from app.core.config import settings
from app.llm.base import LLMClient, LLMResponse


class GeminiClient(LLMClient):
    provider = "gemini"
    default_model = "gemini-3.5-flash-lite" 

    def __init__(self, api_key: str | None = None) -> None:
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        model = model or self.default_model

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        started = time.perf_counter()
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        usage = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            provider=self.provider,
            model=model,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count or 0,
            latency_ms=latency_ms,
        )