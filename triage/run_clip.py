"""Send one clip to the model and print the reply exactly as it comes back.

Run: python -m triage.run_clip            (Ana's clip)
     python -m triage.run_clip yt_c0019   (any clip_id)
"""

import sys

from triage.client import call_model, make_client, response_cost
from triage.config import load_settings
from triage.loader import ANA_CLIP_TITLE, get_clip_by_title, load_clips
from triage.prompt import build_messages


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    clips = load_clips()
    if len(sys.argv) > 1:
        clip = clips[clips["clip_id"] == sys.argv[1]].iloc[0]
    else:
        clip = get_clip_by_title(clips, ANA_CLIP_TITLE)

    settings = load_settings()
    response = call_model(
        make_client(settings),
        build_messages(clip),
        model=settings.model_id,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        langsmith_extra={"metadata": {"clip_id": clip["clip_id"]}},
    )

    cost = response_cost(response)
    print(f"clip_id: {clip['clip_id']}")
    print(f"model: {response.model}")
    print(f"response id: {response.id}")
    print(f"finish_reason: {response.choices[0].finish_reason}")
    print(f"tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
    print(f"cost (billed by OpenRouter): {'unknown' if cost is None else f'${cost:.6f}'}")
    print(f"langsmith: {'project ' + settings.langsmith_project if settings.tracing else 'off (no LANGSMITH_API_KEY)'}")
    print("----- raw model reply (unedited) -----")
    # Written as-is: no strip, no JSON parsing, no reformatting.
    sys.stdout.write(response.choices[0].message.content)
    print("\n----- end raw model reply -----")
    return 0


if __name__ == "__main__":
    sys.exit(main())
