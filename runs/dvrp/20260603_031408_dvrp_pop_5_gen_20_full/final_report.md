# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000093` | valid | 31.0238 | depot_return_pressure_stronger |
| 2 | `cand_000080` | valid | 31.0238 | depot_return_penalty_increased |
| 3 | `cand_000088` | valid | 31.0238 | increased_penalty_truck_balance |
| 4 | `cand_000090` | valid | 31.0238 | depot_return_pressure_strong |
| 5 | `cand_000065` | valid | 35.9195 | depot_return_pressure_isolation |

## Search Best Candidate

- ID: `cand_000093`
- Name: depot_return_pressure_stronger
- Status: valid
- Search Gap: 31.02375097447745
- Thought: Strengthen depot-return penalty (weight 1.0) to reduce max completion time; keep isolation and distance-based heuristic.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 36.0362850905051
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 36.0362850905051
- Penalized mean gap: 36.0362850905051
- Mean TTT: 4.245468049913528
- Penalized mean TTT: 4.245468049913528
- Timeout penalty: 0.0
- Median gap: 34.86607035958948
- Worst gap: 89.33696561190621
- Best gap: -11.25272007104639
- Gap by instance size: {'10': 25.34744249212496, '100': 44.83565192334342, '20': 25.26300647621215, '200': 42.25307831875429, '50': 42.4822462420907}
- TTT by instance size: {'10': 3.8803781248701, '100': 4.083900703613031, '20': 5.146880539543749, '200': 4.07783505211441, '50': 4.038345829426354}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 20
- Feedback calls: 20
- Total API calls: 125
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 20
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 18 / 30
- History buckets: 13
- Added candidates: 83
- Duplicate rejections: 2
- History parent selections: 67
- History offspring with history parent: 50
- Final selection from history: False
