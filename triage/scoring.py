"""Score a run's results against the ground truth and the brief's targets.

A result is one dict per clip as written by run_all: clip_id, raw_reply,
decision (parsed dict or None), parse_error, cost, tokens.
"""

import json
from pathlib import Path

import pandas as pd

from triage.loader import ANA_CLIP_TITLE

SEVERITY = {"keep": 0, "age_restrict": 1, "remove": 2}
CALLS = list(SEVERITY)

TARGETS = {
    "harmful_recall": 0.90,  # at least
    "too_harsh_rate": 1 / 30,  # at most
    "max_cost_usd": 0.05,  # below
    "schema_pass_rate": 1.0,  # exactly
}


def load_results(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_segments(cell: str) -> list[tuple[float, float]]:
    return [tuple(float(x) for x in seg.split("-")) for seg in cell.split(";") if seg.strip()]


def overlaps(start: float, end: float, segments: list[tuple[float, float]]) -> bool:
    return any(start < seg_end and end > seg_start for seg_start, seg_end in segments)


def _joined(results: list[dict], clips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    truth = clips.set_index("clip_id")
    for r in results:
        clip = truth.loc[r["clip_id"]]
        decision = r.get("decision")
        rows.append({
            "clip_id": r["clip_id"],
            "title": clip["title"],
            "is_news": str(clip["is_news"]).lower() == "true",
            "duration": float(clip["duration_seconds"]),
            "truth": clip["ground_truth_label"],
            "segments": parse_segments(clip["ground_truth_violent_segments"]),
            "pred": decision["call"] if decision else None,
            "evidence": decision["evidence"] if decision else [],
            "cost": r.get("cost"),
        })
    return pd.DataFrame(rows)


def score(results: list[dict], clips: pd.DataFrame) -> dict:
    df = _joined(results, clips)
    parsed = df[df["pred"].notna()]

    harmful = df[df["truth"] == "remove"]
    harmful_caught = harmful[harmful["pred"] == "remove"]

    too_harsh = parsed[parsed["pred"].map(SEVERITY) > parsed["truth"].map(SEVERITY)]
    too_lenient = parsed[parsed["pred"].map(SEVERITY) < parsed["truth"].map(SEVERITY)]
    news = df[df["is_news"]]
    news_harsh = too_harsh[too_harsh["is_news"]]

    evidence = [(e, row.duration) for row in parsed.itertuples() for e in row.evidence]
    inside = sum(0 <= e["start"] <= e["end"] <= dur for e, dur in evidence)

    # Unparsed replies count as no overlap rather than being dropped.
    violent = df[df["segments"].map(len) > 0]
    violent_hit = sum(
        any(overlaps(e["start"], e["end"], row.segments) for e in row.evidence) for row in violent.itertuples()
    )

    costs = df["cost"].dropna()
    confusion = pd.crosstab(
        df["truth"], df["pred"].fillna("unparsed"), rownames=["truth"], colnames=["pred"]
    ).reindex(index=CALLS, fill_value=0)

    return {
        "n_clips": len(df),
        "accuracy": float((df["pred"] == df["truth"]).mean()),
        "harmful_recall": len(harmful_caught) / len(harmful) if len(harmful) else 1.0,
        "harmful_missed": harmful[harmful["pred"] != "remove"][["clip_id", "title", "truth", "pred"]].to_dict("records"),
        "too_harsh_rate": len(too_harsh) / len(df),
        "too_harsh": too_harsh[["clip_id", "title", "truth", "pred"]].to_dict("records"),
        "news_too_harsh_rate": len(news_harsh) / len(news) if len(news) else 0.0,
        "too_lenient": too_lenient[["clip_id", "title", "truth", "pred"]].to_dict("records"),
        "schema_pass_rate": len(parsed) / len(df),
        "unparsed": df[df["pred"].isna()]["clip_id"].tolist(),
        "timestamps_inside_rate": inside / len(evidence) if evidence else 1.0,
        "violent_overlap_rate": violent_hit / len(violent) if len(violent) else 1.0,
        "mean_cost_usd": float(costs.mean()) if len(costs) else None,
        "max_cost_usd": float(costs.max()) if len(costs) else None,
        "total_cost_usd": float(costs.sum()) if len(costs) else None,
        "confusion": confusion,
        "ana": ana_check(df, clips),
    }


def ana_check(df: pd.DataFrame, clips: pd.DataFrame) -> dict:
    ana_id = clips.loc[clips["title"] == ANA_CLIP_TITLE, "clip_id"].iloc[0]
    rows = df[df["clip_id"] == ana_id]
    if rows.empty:
        return {"clip_id": ana_id, "ran": False, "passed": False}
    row = rows.iloc[0]
    has_evidence = len(row["evidence"]) > 0
    hit = any(overlaps(e["start"], e["end"], row["segments"]) for e in row["evidence"])
    return {
        "clip_id": ana_id,
        "ran": True,
        "pred": row["pred"],
        "has_timestamps": has_evidence,
        "overlaps_violent": hit,
        "passed": row["pred"] == "age_restrict" and has_evidence and hit,
    }


def target_status(s: dict) -> dict[str, bool]:
    return {
        "harmful_recall": s["harmful_recall"] >= TARGETS["harmful_recall"],
        "too_harsh_rate": s["too_harsh_rate"] <= TARGETS["too_harsh_rate"],
        "max_cost_usd": s["max_cost_usd"] is not None and s["max_cost_usd"] < TARGETS["max_cost_usd"],
        "schema_pass_rate": s["schema_pass_rate"] >= TARGETS["schema_pass_rate"],
        "ana": s["ana"]["passed"],
    }


def _clip_list(rows: list[dict]) -> str:
    if not rows:
        return "- none\n"
    return "".join(f"- `{r['clip_id']}` {r['title']}: truth **{r['truth']}**, model **{r['pred']}**\n" for r in rows)


def summary_markdown(s: dict, run_label: str, model_id: str) -> str:
    status = target_status(s)
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    ana = s["ana"]
    lines = [
        f"# Scorecard: {run_label}",
        "",
        f"Model `{model_id}`, {s['n_clips']} clips, overall accuracy {s['accuracy']:.1%}.",
        "",
        "| Target | Result | Goal | Status |",
        "|---|---|---|---|",
        f"| Harmful clips caught (truth remove → model remove) | {s['harmful_recall']:.1%} | ≥ 90% | {mark(status['harmful_recall'])} |",
        f"| Too harsh (call heavier than truth), all clips | {s['too_harsh_rate']:.1%} | ≤ 3.3% (1 in 30) | {mark(status['too_harsh_rate'])} |",
        f"| Max cost per clip (billed by OpenRouter) | ${s['max_cost_usd']:.4f} | < $0.05 | {mark(status['max_cost_usd'])} |",
        f"| Replies passing Pydantic schema | {s['schema_pass_rate']:.1%} | 100% | {mark(status['schema_pass_rate'])} |",
        f"| Ana's clip ({ana['clip_id']}) age_restrict with overlapping timestamps | {ana.get('pred')} | age_restrict | {mark(status['ana'])} |",
        "",
        "Supporting numbers:",
        f"- Too harsh on news clips only: {s['news_too_harsh_rate']:.1%}",
        f"- Evidence timestamps inside clip duration: {s['timestamps_inside_rate']:.1%}",
        f"- Clips with labelled violent segments where evidence overlaps one: {s['violent_overlap_rate']:.1%}",
        f"- Mean cost per clip ${s['mean_cost_usd']:.4f}, total run ${s['total_cost_usd']:.4f}",
        "",
        "## Confusion matrix (rows = truth, columns = model)",
        "",
        s["confusion"].to_markdown(),
        "",
        "## Harmful clips missed",
        _clip_list(s["harmful_missed"]),
        "## Too harsh",
        _clip_list(s["too_harsh"]),
        "## Too lenient (includes age_restrict called keep)",
        _clip_list(s["too_lenient"]),
        "## Unparsed replies",
        "".join(f"- `{c}`\n" for c in s["unparsed"]) or "- none\n",
        "Interpretation: \"harmful\" = ground truth `remove`. \"Too harsh\" = model call more severe than "
        "ground truth on the keep < age_restrict < remove scale, over all clips.",
        "",
    ]
    return "\n".join(lines)
