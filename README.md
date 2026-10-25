# Zero-google-challenge
it is a challengE on "ZERO" for google

## Verdict on Ana's protest clip (yt_c0041)

**Removal:** Not removed. The model returned `age_restrict`, which matches the ground truth, where the old sorter removed the clip outright.

**Timestamps:** All 5 evidence timestamps sit inside the 254 s clip. 2 of them (60–95 s and 95–110 s) overlap the labelled violent moment at 55–130 s. The other 3 (0–20 s, 145–185 s and 185–250 s) point to the reporting context and the baton charge outside that window.

Raw model reply, unedited: [outputs/ana_raw_reply.txt](outputs/ana_raw_reply.txt)

## Full test set: all 80 labelled clips

All five targets pass: 100% of harmful clips caught, 0% too harsh, max $0.0074 per clip, 100% schema pass, Ana's clip age_restrict with overlapping timestamps. Overall accuracy is 97.5%. The 2 misses are derby-disorder news clips called keep instead of age_restrict.

Scorecard: [outputs/runs/20261025T230748Z/summary.md](outputs/runs/20261025T230748Z/summary.md). Raw replies per clip: [results.jsonl](outputs/runs/20261025T230748Z/results.jsonl)

Caveat: the prompt's policy rules were written with this test set visible, so these scores may be optimistic on unseen clips.

Reproduce with `python -m triage.run_all`, which costs about $0.35. Then run `python -m pytest`, which scores the latest full run.
