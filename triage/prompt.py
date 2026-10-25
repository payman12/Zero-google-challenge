"""Build the model prompt for one clip from the three allowed fields only.

Ground-truth columns never reach the model: build_messages reads PROMPT_FIELDS
by name and nothing else from the row.
"""

import pandas as pd

PROMPT_FIELDS = ("transcript_timestamped", "frame_captions_timestamped", "duration_seconds")

SYSTEM_PROMPT = """\
You are a video safety reviewer. You read a clip's timed transcript and timed frame \
captions and choose one call:

- keep: fine for all viewers.
- age_restrict: graphic or distressing, but has news, documentary, educational or \
safety value, or is fictional/game violence that is extreme. Adults may watch it.
- remove: violates policy whatever the audience.

Before deciding, work out WHY the footage was posted. The same violent moment can be \
reporting or entertainment. Look at how the speaker frames it: context, sourcing, \
warnings and public-interest framing point to reporting; cheering, mocking, "part 2", \
"share this", emojis and celebration point to glorification.

Remove when the clip:
- shares real-world violence to glorify, celebrate or entertain;
- threatens or incites violence against an identifiable person or group;
- promotes or instructs self-harm or suicide;
- exposes victims, especially minors, to harassment (names, schools, faces flagged).

Age-restrict when the clip shows graphic injury, blood, or force against people but \
the purpose is news, documentary, safety education or an appeal for witnesses; also \
for extreme gore in games, and for weapon handling content.

Keep when violence is absent, not graphic, fictional and mild, or part of a regulated \
sport or instruction (boxing, MMA, self-defence technique), and for ordinary news \
that describes violence without showing it.

Being too harsh has a cost: removing legitimate news is censorship. When a clip has \
real news or documentary purpose, prefer age_restrict over remove. But never let \
harmful glorification through as keep.

Evidence: list the moments that drove your call as start and end times in SECONDS \
from the start of the clip (convert [mm:ss] stamps; 01:35 = 95). Each moment needs \
the source (transcript or frames) and a short reason. For keep with nothing \
concerning, evidence may be empty.

Reply with JSON only, no prose, in exactly this shape:
{"call": "keep" | "age_restrict" | "remove",
 "purpose": "<one sentence: why this footage was posted>",
 "evidence": [{"start": <seconds>, "end": <seconds>, "source": "transcript" | "frames", "reason": "<short>"}]}
"""


def _segments(cell: str, empty_note: str) -> str:
    cell = cell.strip()
    if not cell:
        return empty_note
    return "\n".join(segment.strip() for segment in cell.split(" | "))


def build_user_prompt(clip: pd.Series) -> str:
    fields = {name: clip[name] for name in PROMPT_FIELDS}
    transcript = _segments(str(fields["transcript_timestamped"]), "(no speech in this clip)")
    frames = _segments(str(fields["frame_captions_timestamped"]), "(no frame captions)")
    return (
        f"Clip duration: {float(fields['duration_seconds']):.1f} seconds\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Frame captions:\n{frames}"
    )


def build_messages(clip: pd.Series) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(clip)},
    ]


if __name__ == "__main__":
    from triage.loader import ANA_CLIP_TITLE, get_clip_by_title, load_clips

    ana = get_clip_by_title(load_clips(), ANA_CLIP_TITLE)
    print(build_user_prompt(ana))
