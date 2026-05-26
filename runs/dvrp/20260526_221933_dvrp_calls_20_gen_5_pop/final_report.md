# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000080` | valid | 28.5514 | simple_cooperative_depot_wait |
| 2 | `cand_000103` | valid | 29.0413 | adaptive_depot_coefficient_with_isolation_wait |
| 3 | `cand_000104` | valid | 29.0413 | adaptive_competition_with_improved_wait |
| 4 | `cand_000095` | valid | 29.697 | fleet_state_depot |
| 5 | `cand_000097` | valid | 30.8129 | adaptive_depot_coefficient_mutation |

## Search Best Candidate

- ID: `cand_000080`
- Name: simple_cooperative_depot_wait
- Status: valid
- Search Gap: 28.55142472417224
- Thought: Dispatch rule: score = dist_to_customer - nearest_other_truck_dist + beta * dist_to_depot; wait if current truck is much farther than all other trucks for all customers. Uses fixed beta=0.3.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 32.63793219677576
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 32.63793219677576
- Penalized mean gap: 32.63793219677576
- Mean TTT: 4.146014707582891
- Penalized mean TTT: 4.146014707582891
- Timeout penalty: 0.0
- Median gap: 31.930608932936074
- Worst gap: 79.92805571004953
- Best gap: -15.53685033724844
- Gap by instance size: {'10': 25.34744249212496, '100': 39.351991993876446, '20': 25.26300647621215, '200': 31.181542389383672, '50': 42.045677632281546}
- TTT by instance size: {'10': 3.8803781248701, '100': 3.9210109688236523, '20': 5.146880539543749, '200': 3.7592687462808696, '50': 4.0225351583960824}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 32
- Feedback calls: 32
- Total API calls: 137
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 32
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 21 / 64
- History buckets: 12
- Added candidates: 92
- Duplicate rejections: 3
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
