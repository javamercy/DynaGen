# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000102` | valid | 34.1369 | regret_balanced_wait_tuned_v3 |
| 2 | `cand_000099` | valid | 34.2728 | regret_balanced_wait_tuned_v2 |
| 3 | `cand_000037` | valid | 34.4314 | regret_balanced_wait |
| 4 | `cand_000105` | valid | 34.4314 | regret_balanced_wait_tuned2 |
| 5 | `cand_000088` | valid | 34.7018 | regret_balanced_wait_tuned |

## Search Best Candidate

- ID: `cand_000102`
- Name: regret_balanced_wait_tuned_v3
- Status: valid
- Search Gap: 34.1369305845709
- Thought: Reduce waiting threshold to max(2, n_trucks) and increase depot reduction beta to 0.3 to encourage earlier dispatch and stronger depot-return pressure.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 39.97751387404232
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 39.97751387404232
- Penalized mean gap: 39.97751387404232
- Mean TTT: 4.4068403353137855
- Penalized mean TTT: 4.4068403353137855
- Timeout penalty: 0.0
- Median gap: 39.219388442371944
- Worst gap: 83.13439082073859
- Best gap: -23.289673603323187
- Gap by instance size: {'10': 37.95351214618832, '100': 40.11518743142897, '20': 42.34868822414605, '200': 40.1099933517562, '50': 39.36018821669207}
- TTT by instance size: {'10': 4.28783336161052, '100': 3.9442828614035514, '20': 5.837718742016197, '200': 4.010158169625462, '50': 3.9542085419131956}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 31
- Feedback calls: 31
- Total API calls: 136
- Failed calls: 1
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 31
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 14 / 64
- History buckets: 12
- Added candidates: 88
- Duplicate rejections: 9
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
