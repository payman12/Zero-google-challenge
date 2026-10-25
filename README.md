# Zero-google-challenge
it is a challengE on "ZERO" for google

## Verdict on Ana's protest clip (yt_c0041)

**Removal:** Not removed. The model returned `age_restrict`, which matches the ground truth, where the old sorter removed the clip outright.

**Timestamps:** All 5 evidence timestamps sit inside the 254 s clip. 2 of them fall inside the labelled violent window at 55–130 s: the range 55–95 s (tear gas, transcript) and the single point 110 s (the injured-leg frame, start = end = 110, 20 s before the window ends). The other 3 are outside it: the range 0–20 s (journalist credentials), the single point 145 s (the baton-charge frame, 15 s after the window ends) and the range 185–230 s (ministry, ombudsman, rushes).

**Right or wrong:** Right. Age-restrict is the call the labels expect ("Graphic footage of injuries during a protest dispersal — age-restrict, news value"). The injured-leg frame and baton charges are too graphic to keep for teenage viewers. But the model correctly read the purpose as documentary reporting on alleged excessive force, not glorification ("graphic but documentary"), and that news value is what rules out removal.

Raw model reply, unedited: [outputs/ana_raw_reply_v2.txt](outputs/ana_raw_reply_v2.txt). An earlier run with the previous prompt, before single frames were given as points, is in [outputs/ana_raw_reply.txt](outputs/ana_raw_reply.txt). It made the same call and also had 2 of 5 spans in the window: the ranges 60–95 s and 95–110 s.

## Full test set: all 80 labelled clips

All five targets pass: 100% of harmful clips caught, 0% too harsh, max $0.0074 per clip, 100% schema pass, Ana's clip age_restrict with overlapping timestamps. Overall accuracy is 97.5%. The 2 misses are derby-disorder news clips called keep instead of age_restrict.

Scorecard: [outputs/runs/20261025T230748Z/summary.md](outputs/runs/20261025T230748Z/summary.md). Raw replies per clip: [results.jsonl](outputs/runs/20261025T230748Z/results.jsonl)

Caveat: the prompt's policy rules were written with this test set visible, so these scores may be optimistic on unseen clips.

Reproduce with `python -m triage.run_all`, which costs about $0.35. Then run `python -m pytest`, which scores the latest full run.
