# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000087` | valid | 30.0477 | adaptive_fleet_aware_wait |
| 2 | `cand_000080` | valid | 30.7352 | responsive_imbalance_regret |
| 3 | `cand_000066` | valid | 30.7352 | truck_aware_regret_threshold |
| 4 | `cand_000099` | valid | 30.7352 | fleet_imbalance_regret_adaptive |
| 5 | `cand_000075` | valid | 30.7352 | simplified_regret_adaptive |

## Search Best Candidate

- ID: `cand_000087`
- Name: adaptive_fleet_aware_wait
- Status: valid
- Search Gap: 30.04766256079111
- Thought: Uses regret-based dispatch; if no negative regret, applies adaptive waiting threshold that incorporates both current truck's distance to depot and the maximum distance among all trucks to depot. Threshold = alpha * (current_dist_to_depot + max_fleet_dist_to_depot) / 2, where alpha scales with customer density per truck (0.1-0.5). This encourages serving when fleet is far and penalizes waiting when current is far. Tie-break by regret then immediate total distance.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 37.14374769994817
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 37.14374769994817
- Penalized mean gap: 37.14374769994817
- Mean TTT: 4.326494780325303
- Penalized mean TTT: 4.326494780325303
- Timeout penalty: 0.0
- Median gap: 36.88144641433567
- Worst gap: 83.13439082073859
- Best gap: -21.517185189511324
- Gap by instance size: {'10': 37.95351214618832, '100': 35.42402033408333, '20': 42.34868822414605, '200': 30.41072926099086, '50': 39.5817885343323}
- TTT by instance size: {'10': 4.28783336161052, '100': 3.81601941409065, '20': 5.837718742016197, '200': 3.7357514004311376, '50': 3.9551509834780108}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 22
- Feedback calls: 22
- Total API calls: 127
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 22
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 15 / 30
- History buckets: 12
- Added candidates: 64
- Duplicate rejections: 0
- History parent selections: 67
- History offspring with history parent: 50
- Final selection from history: False
