# YouTube Clip Triage: flag harmful videos without silencing the news

An AI triage step that sorts each clip into **keep**, **age-restrict** or **remove**. It decides by asking *why the footage was posted*, not just *what it shows*, and it backs every call with timestamps that point to the moments that drove it.

## The problem

The current sorter scans titles, transcripts and frame descriptions for violent signals. It never asks why a clip was posted, so it can't tell a news report from a clip that glorifies violence.

- **News gets censored.** A journalist's protest footage is removed outright when age-restriction was the right call. Each wrong removal means a reviewer appeal, lost trust with news partners, and accusations of censorship.
- **Harm gets through.** Harmful clips without obvious keywords slip past and reach teenage viewers.
- **Reviewers work blind.** Decisions come with no evidence, so reviewers scrub through whole videos to check a call.

## The solution

For each clip, a language model reads only three things: the timed transcript, the timed frame descriptions and the clip length. It then:

1. **Works out the purpose.** Is this reporting, education, sport or entertainment? Does it glorify the violence?
2. **Makes the call.** Keep, age-restrict or remove. When a clip has real news value, it prefers age-restrict over remove.
3. **Shows its evidence.** It returns timestamps for the moments behind the decision, so a reviewer can jump straight to them.

Every reply is checked against a strict format before it's accepted. The correct answers in the test data are never shown to the model; automated tests check this on every change.

## Example: Ana's protest clip

`yt_c0041`, "Tear gas on Avenida Central". An accredited freelance journalist films police firing tear gas into a permitted march. The footage includes an injured leg and a baton charge. **The old sorter removed it.**

**Removal:** Not removed. The model returned `age_restrict` ([raw reply](outputs/ana_raw_reply_v2.txt)).

**Timestamps:** All 5 are inside the 254 s clip. 2 fall in the 55–130 s violent window: the range 55–95 s and the point 110 s. 3 do not: the range 0–20 s, the point 145 s and the range 185–230 s.

**My judgment:** both verdicts are right. Not removing it is right, because the model read the purpose as documentary reporting on alleged excessive force ("graphic but documentary"), not glorification. Not keeping it is right, because the injured leg and baton charges are too graphic for teenage viewers. This agrees with the labelled data for `yt_c0041` in [data/clips.csv](data/clips.csv): `ground_truth_label` = `age_restrict`, `ground_truth_policy_reason` = "Graphic footage of injuries during a protest dispersal — age-restrict, news value".

**What to improve:** the 3 timestamps outside the window point to context (the journalist's credentials and the ministry and ombudsman response) and to the baton charge, which falls after the labelled window ends. They explain the call, but they aren't violent moments, so evidence should be split into "what is graphic" and "why it has news value".

An earlier run, made before the prompt told the model to give single frames as points, is in [outputs/ana_raw_reply.txt](outputs/ana_raw_reply.txt). It made the same call, with 2 of 5 spans in the window.

## Results on all 80 labelled clips

| Target | Result | Goal |
|---|---|---|
| Harmful clips caught (should be removed, was removed) | **100%** (16 of 16) | at least 90% |
| Calls harsher than deserved | **0%** (0 of 80) | at most 1 in 30 |
| Cost per clip | **under 1¢** (max $0.0074) | under 5¢ |
| Replies in the valid format, with timestamps | **100%** | 100% |
| Ana's clip age-restricted with timestamps | **yes** | yes |

Overall, 78 of 80 calls (97.5%) match the labels, and the full run cost $0.34. Details: [scorecard](outputs/runs/20261025T230748Z/summary.md) and [raw reply for every clip](outputs/runs/20261025T230748Z/results.jsonl).

**Known limits**
- **The 2 misses are both too lenient.** Two news clips of football derby disorder were kept when they should have been age-restricted.
- **The test set was visible while writing the prompt,** so these scores may be optimistic on new clips. The next step is to test on clips held back and never looked at.
- **This run used 80 clips;** the full set is 200.

## Run it yourself

Requires Python 3.12 and an [OpenRouter](https://openrouter.ai) API key. A [LangSmith](https://smith.langchain.com) key is optional and adds cost tracing.

```bash
cp .env.example .env              # then paste your keys into .env
python -m triage.run_clip         # Ana's clip, prints the raw reply (~0.6¢)
python -m triage.run_all          # all 80 clips + scorecard (~$0.35)
python -m pytest                  # scores the latest run against the targets (free)
```

The model is set in [config.toml](config.toml) and can be swapped without code changes.

## What's in the repo

| Path | What it does |
|---|---|
| [triage/prompt.py](triage/prompt.py) | The instructions and the clip data sent to the model |
| [triage/schema.py](triage/schema.py) | The required reply format: call, purpose, evidence timestamps |
| [triage/run_all.py](triage/run_all.py) | Runs every clip and saves the raw replies |
| [triage/scoring.py](triage/scoring.py) | Scores the replies against the labelled answers |
| [tests/](tests/) | Checks that the answers never leak into the prompt, plus scoring against the targets |
| [data/clips.csv](data/clips.csv) | The labelled test clips |
