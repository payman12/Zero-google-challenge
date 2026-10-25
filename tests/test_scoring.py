import pytest
from pydantic import ValidationError

from triage.loader import load_clips
from triage.schema import parse_decision
from triage.scoring import overlaps, parse_segments, score, target_status

CLIPS = load_clips()
TRUTH = CLIPS.set_index("clip_id")["ground_truth_label"]


def _result(clip_id, call, evidence=(), cost=0.006):
    decision = {"call": call, "purpose": "test", "evidence": [
        {"start": s, "end": e, "source": "frames", "reason": "r"} for s, e in evidence
    ]}
    return {"clip_id": clip_id, "decision": decision, "cost": cost}


def _perfect_results():
    out = []
    for _, clip in CLIPS.iterrows():
        segs = parse_segments(clip["ground_truth_violent_segments"])
        out.append(_result(clip["clip_id"], clip["ground_truth_label"], segs[:1]))
    return out


# --- schema ---

def test_parse_valid_reply():
    d = parse_decision('{"call":"keep","purpose":"cooking","evidence":[]}')
    assert d.call == "keep" and d.evidence == []


def test_single_frame_instant_is_valid_evidence():
    d = parse_decision('{"call":"remove","purpose":"p","evidence":[{"start":35,"end":35,"source":"frames","reason":"r"}]}')
    assert d.evidence[0].start == d.evidence[0].end == 35
    assert overlaps(35, 35, [(8, 72)])


def test_parse_strips_code_fence_for_parsing():
    raw = '```json\n{"call":"remove","purpose":"p","evidence":[{"start":1,"end":2,"source":"frames","reason":"r"}]}\n```'
    assert parse_decision(raw).call == "remove"


@pytest.mark.parametrize("raw", [
    '{"call":"delete","purpose":"p","evidence":[]}',  # unknown call
    '{"call":"keep","purpose":"","evidence":[]}',  # empty purpose
    '{"call":"keep","purpose":"p","evidence":[{"start":5,"end":2,"source":"frames","reason":"r"}]}',  # start > end
    '{"call":"keep","purpose":"p","evidence":[{"start":-1,"end":2,"source":"frames","reason":"r"}]}',  # negative
    '{"call":"keep","purpose":"p","evidence":[{"start":1,"end":2,"source":"audio","reason":"r"}]}',  # bad source
    '{"call":"keep","purpose":"p"}',  # evidence missing
    "I think this should be kept.",  # not JSON
    "",
])
def test_parse_rejects_bad_replies(raw):
    with pytest.raises(ValidationError):
        parse_decision(raw)


# --- scoring ---

def test_overlap_and_segments():
    assert parse_segments("8.0-72.0;62.0-130.0") == [(8.0, 72.0), (62.0, 130.0)]
    assert parse_segments("") == []
    assert overlaps(60, 95, [(55, 130)])
    assert not overlaps(0, 55, [(55, 130)])  # touching edge is not overlap


def test_perfect_run_passes_every_target():
    s = score(_perfect_results(), CLIPS)
    assert s["accuracy"] == 1.0
    assert all(target_status(s).values())


def test_missed_harmful_clip_lowers_recall():
    results = _perfect_results()
    harmful = [r for r in results if TRUTH[r["clip_id"]] == "remove"]
    harmful[0]["decision"]["call"] = "age_restrict"
    s = score(results, CLIPS)
    assert s["harmful_recall"] == pytest.approx((len(harmful) - 1) / len(harmful))
    assert s["harmful_missed"][0]["clip_id"] == harmful[0]["clip_id"]


def test_removing_a_keep_clip_counts_as_too_harsh():
    results = _perfect_results()
    keeps = [r for r in results if TRUTH[r["clip_id"]] == "keep"]
    for r in keeps[:3]:
        r["decision"]["call"] = "remove"
    s = score(results, CLIPS)
    assert s["too_harsh_rate"] == pytest.approx(3 / len(results))
    assert not target_status(s)["too_harsh_rate"]  # 3/80 > 1/30


def test_unparsed_reply_fails_schema_target():
    results = _perfect_results()
    results[0]["decision"] = None
    s = score(results, CLIPS)
    assert s["unparsed"] == [results[0]["clip_id"]]
    assert not target_status(s)["schema_pass_rate"]


def test_expensive_clip_fails_cost_target():
    results = _perfect_results()
    results[5]["cost"] = 0.051
    assert not target_status(score(results, CLIPS))["max_cost_usd"]


def test_ana_removed_fails_ana_check():
    results = _perfect_results()
    for r in results:
        if r["clip_id"] == "yt_c0041":
            r["decision"]["call"] = "remove"
    assert not score(results, CLIPS)["ana"]["passed"]


def test_ana_without_timestamps_fails_ana_check():
    results = _perfect_results()
    for r in results:
        if r["clip_id"] == "yt_c0041":
            r["decision"]["evidence"] = []
    assert not score(results, CLIPS)["ana"]["passed"]
