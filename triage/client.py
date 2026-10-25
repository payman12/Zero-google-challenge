"""OpenRouter client. OpenRouter speaks the OpenAI API, so the OpenAI SDK is reused."""

from openai import OpenAI

from triage.config import Settings


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)
