"""Smoke test for step 2: config readable, key present, model slug valid, one call works.

Run: python -m triage.check_setup
"""

import sys

import httpx

from triage.client import make_client
from triage.config import load_settings


def main() -> int:
    settings = load_settings(require_key=False)
    print(f"model id (from config.toml): {settings.model_id}")

    # The model list is public, so the slug can be checked before a key exists.
    models = httpx.get(f"{settings.base_url}/models", timeout=30).json()["data"]
    model = next((m for m in models if m["id"] == settings.model_id), None)
    if model is None:
        print(f"FAIL: {settings.model_id!r} is not an OpenRouter model id")
        return 1
    price_in = float(model["pricing"]["prompt"]) * 1_000_000
    price_out = float(model["pricing"]["completion"]) * 1_000_000
    print(f"slug valid: ${price_in:.2f}/M input, ${price_out:.2f}/M output")

    if not settings.api_key:
        print("FAIL: OPENROUTER_API_KEY is empty: paste your key into .env")
        return 1
    print(f"key loaded from .env: ...{settings.api_key[-4:]}")

    client = make_client(settings)
    response = client.chat.completions.create(
        model=settings.model_id,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=10,
        temperature=settings.temperature,
    )
    usage = response.usage
    cost = (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1_000_000
    print(f"reply: {response.choices[0].message.content!r}")
    print(f"tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out, ~${cost:.6f}")
    print("OK: setup works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
