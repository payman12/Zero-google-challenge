"""Send one clip to the model and print the reply exactly as it comes back.

Run: python -m triage.run_clip            (Ana's clip)
     python -m triage.run_clip yt_c0019   (any clip_id)
"""

import sys

from triage.client import make_client
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
    response = make_client(settings).chat.completions.create(
        model=settings.model_id,
        messages=build_messages(clip),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    print(f"clip_id: {clip['clip_id']}")
    print(f"model: {response.model}")
    print(f"response id: {response.id}")
    print(f"finish_reason: {response.choices[0].finish_reason}")
    print(f"tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
    print("----- raw model reply (unedited) -----")
    # Written as-is: no strip, no JSON parsing, no reformatting.
    sys.stdout.write(response.choices[0].message.content)
    print("\n----- end raw model reply -----")
    return 0


if __name__ == "__main__":
    sys.exit(main())
