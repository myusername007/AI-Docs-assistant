# Ціни в USD за 1 000 000 токенів: (input, output).
PRICING: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-sonnet-4-5"): (3.00, 15.00),
    ("openai", "gpt-5.6-luna"): (0.20, 1.20),
    ("gemini", "gemini-2.5-flash"): (0.30, 2.50),
}


def _lookup(provider: str, model: str) -> tuple[float, float] | None:
    if (provider, model) in PRICING:
        return PRICING[(provider, model)]
    # сервер часто повертає модель із суфіксом дати -> збіг за префіксом
    for (p, m), price in PRICING.items():
        if p == provider and model.startswith(m):
            return price
    return None


def estimate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    prices = _lookup(provider, model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000