import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return ENCODING.decode(tokens[:max_tokens])


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = {
        "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
        "text-embedding-3-small": {"input": 0.020 / 1_000_000, "output": 0},
    }
    p = pricing.get(model, pricing["gpt-4o"])
    return (input_tokens * p["input"]) + (output_tokens * p["output"])