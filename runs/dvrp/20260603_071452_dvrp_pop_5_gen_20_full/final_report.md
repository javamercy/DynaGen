# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000079` | valid | 28.9321 | regret_with_far_truck_penalty |
| 2 | `cand_000044` | valid | 29.5081 | adaptive_threshold_savings |
| 3 | `cand_000054` | valid | 30.0428 | adaptive_threshold_with_distance_modulation |
| 4 | `cand_000055` | valid | 30.5018 | savings_with_fallback_simplified |
| 5 | `cand_000040` | valid | 30.5018 | savings_with_fallback |

## Search Best Candidate

- ID: `cand_000079`
- Name: regret_with_far_truck_penalty
- Status: valid
- Search Gap: 28.9320826371496
- Thought: Improve TTT by a regret-based rule that penalizes assigning customers to trucks far from depot. For each customer, if the active truck is the best (min cost), select the best savings. Otherwise, accept a fallback if the active cost is within a threshold that increases with the distance of the best other truck from depot, protecting far trucks from additional work. Return None if no acceptable customer.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 37.86796908184104
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 37.86796908184104
- Penalized mean gap: 37.86796908184104
- Mean TTT: 4.346429557873633
- Penalized mean TTT: 4.346429557873633
- Timeout penalty: 0.0
- Median gap: 37.168207330380326
- Worst gap: 83.13439082073859
- Best gap: -19.09555776996516
- Gap by instance size: {'10': 37.95351214618832, '100': 36.75097604513616, '20': 42.34868822414605, '200': 31.376294708057596, '50': 40.91037428567705}
- TTT by instance size: {'10': 4.28783336161052, '100': 3.8476243585223635, '20': 5.837718742016197, '200': 3.7643540099365866, '50': 3.994617317282499}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 28
- Feedback calls: 28
- Total API calls: 133
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 28
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 18 / 30
- History buckets: 12
- Added candidates: 75
- Duplicate rejections: 0
- History parent selections: 67
- History offspring with history parent: 50
- Final selection from history: False
