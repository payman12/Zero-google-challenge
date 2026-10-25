# Scorecard: 20261025T230748Z

Model `anthropic/claude-sonnet-5.5`, 80 clips, overall accuracy 97.5%.

| Target | Result | Goal | Status |
|---|---|---|---|
| Harmful clips caught (truth remove → model remove) | 100.0% | ≥ 90% | PASS |
| Too harsh (call heavier than truth), all clips | 0.0% | ≤ 3.3% (1 in 30) | PASS |
| Max cost per clip (billed by OpenRouter) | $0.0074 | < $0.05 | PASS |
| Replies passing Pydantic schema | 100.0% | 100% | PASS |
| Ana's clip (yt_c0041) age_restrict with overlapping timestamps | age_restrict | age_restrict | PASS |

Supporting numbers:
- Too harsh on news clips only: 0.0%
- Evidence timestamps inside clip duration: 100.0%
- Clips with labelled violent segments where evidence overlaps one: 100.0%
- Mean cost per clip $0.0043, total run $0.3434

## Confusion matrix (rows = truth, columns = model)

| truth        |   age_restrict |   keep |   remove |
|:-------------|---------------:|-------:|---------:|
| keep         |              0 |     44 |        0 |
| age_restrict |             18 |      2 |        0 |
| remove       |              0 |      0 |       16 |

## Harmful clips missed
- none

## Too harsh
- none

## Too lenient (includes age_restrict called keep)
- `yt_c0060` Derby day disorder outside Kingsway Stadium: truth **age_restrict**, model **keep**
- `yt_c0072` Seats thrown as derby match descends into disorder: truth **age_restrict**, model **keep**

## Unparsed replies
- none

Interpretation: "harmful" = ground truth `remove`. "Too harsh" = model call more severe than ground truth on the keep < age_restrict < remove scale, over all clips.
