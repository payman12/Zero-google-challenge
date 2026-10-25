import pytest

from triage.loader import ANA_CLIP_TITLE, get_clip_by_title, load_clips
from triage.prompt import PROMPT_FIELDS, build_messages

GROUND_TRUTH_COLUMNS = [
    "ground_truth_label",
    "ground_truth_violent_segments",
    "ground_truth_policy_reason",
]

CLIPS = load_clips()


def _prompt_text(clip) -> str:
    return "\n".join(m["content"] for m in build_messages(clip))


def test_prompt_fields_exclude_ground_truth():
    assert not set(PROMPT_FIELDS) & set(GROUND_TRUTH_COLUMNS)


@pytest.mark.parametrize("clip_id", CLIPS["clip_id"])
def test_no_ground_truth_reaches_prompt(clip_id):
    clip = CLIPS[CLIPS["clip_id"] == clip_id].iloc[0]
    text = _prompt_text(clip)
    for column in GROUND_TRUTH_COLUMNS:
        assert column not in text
    for value in (clip["ground_truth_policy_reason"], clip["ground_truth_violent_segments"]):
        if value:
            assert value not in text


@pytest.mark.parametrize("clip_id", CLIPS["clip_id"])
def test_prompt_survives_ground_truth_removal(clip_id):
    # If the prompt only reads the allowed fields, dropping every other column changes nothing.
    clip = CLIPS[CLIPS["clip_id"] == clip_id].iloc[0]
    assert _prompt_text(clip[list(PROMPT_FIELDS)]) == _prompt_text(clip)


@pytest.mark.parametrize("clip_id", CLIPS["clip_id"])
def test_only_allowed_fields_reach_prompt(clip_id):
    # Overwrite every non-allowed column with a marker; none may show up and the prompt must not change.
    clip = CLIPS[CLIPS["clip_id"] == clip_id].iloc[0]
    tampered = clip.copy()
    for column in clip.index.difference(PROMPT_FIELDS):
        tampered[column] = f"LEAK_{column}"
    text = _prompt_text(tampered)
    assert "LEAK_" not in text
    assert text == _prompt_text(clip)


def test_ana_prompt_has_timed_transcript_frames_and_duration():
    text = _prompt_text(get_clip_by_title(CLIPS, ANA_CLIP_TITLE))
    assert "Clip duration: 254.0 seconds" in text
    assert "[00:55-01:35] (coughing) First canisters came in here" in text
    assert "[01:50] An injured leg with a visible wound" in text
    assert "age_restrict" in text and '"evidence"' in text


def test_empty_transcript_is_marked_not_blank():
    clip = CLIPS[CLIPS["clip_id"] == "yt_c0007"].iloc[0]
    assert "(no speech in this clip)" in _prompt_text(clip)
