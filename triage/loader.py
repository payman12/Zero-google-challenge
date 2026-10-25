"""Load the labelled clip test set and pick out individual clips by real fields."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clips.csv"

# Ana's protest clip: freelance journalist footage of a tear-gas dispersal.
# Ground truth is age_restrict; the old sorter removed it outright.
ANA_CLIP_TITLE = "Tear gas on Avenida Central: what I filmed from the second line"


def load_clips(path: Path = DATA_PATH) -> pd.DataFrame:
    # keep_default_na=False so empty transcripts stay "" instead of NaN floats
    return pd.read_csv(path, keep_default_na=False, dtype={"is_news": str})


def get_clip_by_title(clips: pd.DataFrame, title: str) -> pd.Series:
    matches = clips[clips["title"] == title]
    if len(matches) != 1:
        raise LookupError(f"expected exactly 1 clip titled {title!r}, found {len(matches)}")
    return matches.iloc[0]


if __name__ == "__main__":
    clips = load_clips()
    print(f"Loaded {len(clips)} clips from {DATA_PATH.name}")

    ana = get_clip_by_title(clips, ANA_CLIP_TITLE)
    transcript = ana["transcript_timestamped"]

    print(f"\n{ana['clip_id']} | {ana['title']}")
    print(f"channel={ana['channel_type']} news={ana['is_news']} label={ana['ground_truth_label']}")
    print(f"segments={ana['ground_truth_violent_segments']}\n")
    print("transcript_timestamped:")
    for segment in transcript.split(" | "):
        print(f"  {segment}")

    assert isinstance(transcript, str) and transcript.strip(), "transcript cell is empty"
    assert transcript.lstrip().startswith("["), "transcript has no timestamps"
    print(f"\nOK: transcript holds {len(transcript)} chars of real timestamped text")
