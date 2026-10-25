"""Score the latest full run against the brief's targets. No API calls: run_all produces the results."""

import pytest

from triage.loader import load_clips
from triage.run_all import RUNS_DIR
from triage.scoring import load_results, score, target_status

CLIPS = load_clips()


def _latest_full_run():
    # Smoke runs (--limit) cover fewer clips and are ignored.
    for run_dir in sorted(RUNS_DIR.glob("*"), reverse=True):
        path = run_dir / "results.jsonl"
        if path.exists() and {r["clip_id"] for r in load_results(path)} >= set(CLIPS["clip_id"]):
            return run_dir
    return None


RUN = _latest_full_run()
pytestmark = pytest.mark.skipif(RUN is None, reason="no full run yet: python -m triage.run_all")
STATUS = target_status(score(load_results(RUN / "results.jsonl"), CLIPS)) if RUN else {}


@pytest.mark.parametrize("target", ["harmful_recall", "too_harsh_rate", "max_cost_usd", "schema_pass_rate", "ana"])
def test_target(target):
    assert STATUS[target], f"{target} failed in {RUN.name}; see {RUN / 'summary.md'}"
