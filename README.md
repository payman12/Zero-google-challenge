# Zero-google-challenge
it is a challengE on "ZERO" for google

## Verdict on Ana's protest clip (yt_c0041)

**Removal:** Not removed. The model returned `age_restrict` ([raw reply](outputs/ana_raw_reply_v2.txt)).

**Timestamps:** All 5 are inside the 254 s clip. 2 fall in the 55–130 s violent window: the range 55–95 s and the point 110 s. 3 do not: the range 0–20 s, the point 145 s and the range 185–230 s.

### Note: my judgment and the labelled data
My judgment is that both verdicts are right. Not removing is right, because the model read the purpose as documentary reporting on alleged excessive force ("graphic but documentary"), not glorification. Not keeping is right, because the injured-leg frame and baton charges are too graphic for teenage viewers. This agrees with the labelled data for yt_c0041 in [data/clips.csv](data/clips.csv): `ground_truth_label` = `age_restrict`, `ground_truth_policy_reason` = "Graphic footage of injuries during a protest dispersal — age-restrict, news value".

An earlier run, made before the prompt told the model to give single frames as points, is in [outputs/ana_raw_reply.txt](outputs/ana_raw_reply.txt). It made the same call and also had 2 of 5 spans in the window: the ranges 60–95 s and 95–110 s.

## Full test set: all 80 labelled clips

All five targets pass: 100% of harmful clips caught, 0% too harsh, max $0.0074 per clip, 100% schema pass, Ana's clip age_restrict with overlapping timestamps. Overall accuracy is 97.5%. The 2 misses are derby-disorder news clips called keep instead of age_restrict.

Scorecard: [outputs/runs/20261025T230748Z/summary.md](outputs/runs/20261025T230748Z/summary.md). Raw replies per clip: [results.jsonl](outputs/runs/20261025T230748Z/results.jsonl)

Caveat: the prompt's policy rules were written with this test set visible, so these scores may be optimistic on unseen clips.

Reproduce with `python -m triage.run_all`, which costs about $0.35. Then run `python -m pytest`, which scores the latest full run.
