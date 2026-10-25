"""Run every clip through the model, save raw replies, and write a scorecard.

Run: python -m triage.run_all                  (all clips, new run folder)
     python -m triage.run_all --limit 3        (smoke test)
     python -m triage.run_all --resume outputs/runs/<run>   (finish a crashed run)
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import RateLimitError
from pydantic import ValidationError

from triage.client import call_model, make_client, response_cost
from triage.config import ROOT, load_settings
from triage.loader import load_clips
from triage.prompt import build_messages
from triage.schema import parse_decision
from triage.scoring import load_results, score, summary_markdown, target_status

RUNS_DIR = ROOT / "outputs" / "runs"


class Pacer:
    """Spaces request starts so the run stays under the per-minute rate limit."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        time.sleep(start - now)


def run_one(client, settings, clip, pacer: Pacer, retries: int = 4) -> dict:
    for attempt in range(retries + 1):
        pacer.wait()
        try:
            response = call_model(
                client,
                build_messages(clip),
                model=settings.model_id,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                langsmith_extra={"metadata": {"clip_id": clip["clip_id"]}},
            )
            break
        except RateLimitError:
            if attempt == retries:
                raise
            time.sleep(20 * (attempt + 1))
    raw = response.choices[0].message.content
    try:
        decision, error = parse_decision(raw or "").model_dump(), None
    except ValidationError as exc:
        decision, error = None, str(exc)
    return {
        "clip_id": clip["clip_id"],
        "raw_reply": raw,
        "decision": decision,
        "parse_error": error,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "cost": response_cost(response),
        "response_id": response.id,
        "finish_reason": response.choices[0].finish_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only run the first N clips")
    parser.add_argument("--resume", type=Path, help="existing run folder to finish")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    settings = load_settings()
    clips = load_clips()
    if args.limit:
        clips = clips.head(args.limit)

    run_dir = args.resume or RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    done = {r["clip_id"] for r in load_results(results_path)} if results_path.exists() else set()
    todo = [clip for _, clip in clips.iterrows() if clip["clip_id"] not in done]
    print(f"run {run_dir.name}: {len(todo)} to call, {len(done)} already done, model {settings.model_id}")
    print(f"langsmith: {'project ' + settings.langsmith_project if settings.tracing else 'off'}")

    client = make_client(settings)
    pacer = Pacer(settings.requests_per_minute)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(results_path, "a", encoding="utf-8") as out:
        futures = {pool.submit(run_one, client, settings, clip, pacer): clip["clip_id"] for clip in todo}
        for i, future in enumerate(as_completed(futures), 1):
            clip_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # network/API failure: leave it for --resume
                print(f"[{i}/{len(todo)}] {clip_id} FAILED: {str(exc)[:160]}")
                continue
            with lock:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
            call = result["decision"]["call"] if result["decision"] else "UNPARSED"
            print(f"[{i}/{len(todo)}] {clip_id} {call} ${result['cost'] or 0:.4f}")

    results = load_results(results_path)
    scorecard = score(results, load_clips())
    (run_dir / "summary.md").write_text(summary_markdown(scorecard, run_dir.name, settings.model_id), encoding="utf-8")

    print()
    if len(results) < len(clips):
        print(f"INCOMPLETE: {len(results)} of {len(clips)} clips done. Finish with:")
        print(f"  python -m triage.run_all --resume {run_dir.relative_to(ROOT).as_posix()}")
        return 1
    for target, ok in target_status(scorecard).items():
        print(f"{'PASS' if ok else 'FAIL'}  {target}")
    print(f"\nscorecard: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
