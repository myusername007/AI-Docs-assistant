import logging

from app.llm.base import LLMClient, LLMResponse
from app.llm.pricing import estimate_cost

logger = logging.getLogger("llm")


class ObservableLLMClient(LLMClient):
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.provider = inner.provider

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = await self._inner.generate(
            prompt, system=system, model=model, max_tokens=max_tokens
        )

        response.cost_usd = estimate_cost(
            response.provider,
            response.model,
            response.input_tokens,
            response.output_tokens,
        )

        logger.info(
            "llm_call",
            extra={
                "provider": response.provider,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": round(response.latency_ms, 1),
                "cost_usd": response.cost_usd,
            },
        )
        return response