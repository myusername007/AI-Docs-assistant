from app.llm.pricing import estimate_cost


def test_exact_match():
    assert estimate_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_prefix_match_for_dated_model():
    assert estimate_cost("anthropic", "claude-sonnet-4-5-20250101", 1_000_000, 0) == 3.0


def test_unknown_model_returns_none():
    assert estimate_cost("openai", "made-up", 100, 100) is None