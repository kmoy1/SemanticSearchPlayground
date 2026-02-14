import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppDefaults:
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 4
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0


DEFAULTS = AppDefaults()

PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
}


def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    model_pricing = PRICING_PER_1M_TOKENS.get(model_name)
    if not model_pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * model_pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output_per_1m"]
    return input_cost + output_cost
