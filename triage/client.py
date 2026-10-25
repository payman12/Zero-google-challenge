"""OpenRouter client. OpenRouter speaks the OpenAI API, so the OpenAI SDK is reused.

call_model is traced to LangSmith (when enabled) with the real cost OpenRouter bills.
"""

from langsmith import traceable
from openai import OpenAI

from triage.config import Settings

TRACED_INPUTS = ("messages", "model", "temperature", "max_tokens")


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)


def _trace_inputs(inputs: dict) -> dict:
    # Whitelist only: the client and settings hold the API key and must never be uploaded.
    return {name: inputs[name] for name in TRACED_INPUTS if name in inputs}


def _trace_outputs(response) -> dict:
    usage = response.usage
    usage_metadata = {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    cost = response_cost(response)
    if cost is not None:
        usage_metadata["total_cost"] = cost
    return {"choices": [c.model_dump() for c in response.choices], "usage_metadata": usage_metadata}


def response_cost(response) -> float | None:
    """USD cost OpenRouter reports for this call (requested via usage.include)."""
    return (response.usage.model_extra or {}).get("cost")


@traceable(
    run_type="llm",
    name="openrouter_chat",
    metadata={"ls_provider": "openrouter"},
    process_inputs=_trace_inputs,
    process_outputs=_trace_outputs,
)
def call_model(client: OpenAI, messages: list[dict], model: str, temperature: float, max_tokens: int):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"usage": {"include": True}},
    )
