# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000035` | valid | 30.2459 | regret_minmax_penalized |
| 2 | `cand_000043` | valid | 30.2459 | penalized_regret |
| 3 | `cand_000098` | valid | 30.2459 | regret_balance_penalty |
| 4 | `cand_000041` | valid | 30.2459 | regret_minmax_penalized |
| 5 | `cand_000048` | valid | 30.2459 | regret_minmax_penalized |

## Search Best Candidate

- ID: `cand_000035`
- Name: regret_minmax_penalized
- Status: valid
- Search Gap: 30.24592198516924
- Thought: Select customer maximizing (regret - 0.1*current_estimated_tour_cost), where regret is the difference between the minimum other truck's estimated cost to serve the customer and the current truck's estimated cost. This penalizes extending the current truck's route, promoting balance. Tie-break by lower current cost. Return None if no customers available.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 42.62745413587081
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 42.62745413587081
- Penalized mean gap: 42.62745413587081
- Mean TTT: 4.481859084471188
- Penalized mean TTT: 4.481859084471188
- Timeout penalty: 0.0
- Median gap: 42.69558567058981
- Worst gap: 83.13439082073859
- Best gap: -8.153272496210347
- Gap by instance size: {'10': 37.95351214618832, '100': 45.007292569268856, '20': 42.34868822414605, '200': 45.267064581187775, '50': 42.56071315856306}
- TTT by instance size: {'10': 4.28783336161052, '100': 4.081123998697474, '20': 5.837718742016197, '200': 4.164629929188836, '50': 4.037989390842916}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 23
- Feedback calls: 23
- Total API calls: 128
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 23
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 15 / 30
- History buckets: 11
- Added candidates: 83
- Duplicate rejections: 5
- History parent selections: 67
- History offspring with history parent: 50
- Final selection from history: False
