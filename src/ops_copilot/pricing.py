"""USD per million tokens, Anthropic first-party rates (October 2026).

Cache writes (5-minute TTL) cost 1.25x base input; cache reads are listed per
model. Keep this table next to the code that uses it so cost-per-run numbers
in traces and eval reports stay honest when prices change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    price = PRICES.get(model)
    if price is None:
        return 0.0
    return (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read * price.cache_read
        + cache_write * price.cache_write
    ) / 1_000_000
