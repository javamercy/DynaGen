# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Distance | Name |
|---:|---|---|---:|---|
| 1 | `cand_000096` | valid | 17.4023 | regret_2opt_double_bridge_restart8 |
| 2 | `cand_000045` | valid | 17.4218 | regret_steepest_doublebridge |
| 3 | `cand_000084` | valid | 17.4315 | regret_2opt_perturb |
| 4 | `cand_000099` | valid | 17.4345 | regret_2opt_3opt_restart |
| 5 | `cand_000072` | valid | 17.439 | regret_2opt_restart |

## Search Best Candidate

- ID: `cand_000096`
- Name: regret_2opt_double_bridge_restart8
- Status: valid
- Search Distance: 17.402310615804943
- Thought: Regret insertion with random start and tie-breaking, then 2-opt with nearest-neighbor candidate list (k=50). If 2-opt stalls, apply a double-bridge kick for diversification. Restart threshold set to budget//8 for more frequent diversification. Maintains early best tour reporting and budget discipline.

## Test Evaluation

- Status: valid
- Test distance: 4.1102674899907585
- Instances evaluated: 10
- Valid runs: 10 / 10
- Scored runs: 10 / 10
- Partial timeout runs: 0
- Mean gap: 4.1102674899907585
- Penalized mean gap: 4.1102674899907585
- Timeout penalty: 0.0
- Median gap: 4.019631073310459
- Worst gap: 7.46082216670452
- Best gap: 1.0773130544993663

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 32
- Feedback calls: 32
- Total API calls: 137
- Failed calls: 1
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
- History size: 15 / 64
- History buckets: 10
- Added candidates: 96
- Duplicate rejections: 1
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
