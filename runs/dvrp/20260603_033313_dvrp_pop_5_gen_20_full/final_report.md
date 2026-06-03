# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000099` | valid | 29.0142 | makespan_regret_with_penalty |
| 2 | `cand_000087` | valid | 30.483 | adaptive_regret_relative_wait |
| 3 | `cand_000100` | valid | 31.2368 | adaptive_regret_std_wait |
| 4 | `cand_000089` | valid | 31.5119 | adaptive_regret_reduced_wait |
| 5 | `cand_000090` | valid | 31.5119 | less_wait_adaptive_regret |

## Search Best Candidate

- ID: `cand_000099`
- Name: makespan_regret_with_penalty
- Status: valid
- Search Gap: 29.0142070437765
- Thought: Regret-based selection with additional makespan urgency: score = (cost_now - second_best_cost) - 0.5 * (worst_cost - cost_now). Dynamic wait threshold increases with number of available customers. If only one truck, use nearest customer by total tour cost.

## Test Evaluation

- Problem: DVRP
- Status: valid
- Test gap: 37.07463528141665
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Mean gap: 37.07463528141665
- Penalized mean gap: 37.07463528141665
- Mean TTT: 4.323963297833875
- Penalized mean TTT: 4.323963297833875
- Timeout penalty: 0.0
- Median gap: 36.278577920873545
- Worst gap: 83.13439082073859
- Best gap: -14.234574507979922
- Gap by instance size: {'10': 37.95351214618832, '100': 35.46535471189657, '20': 42.34868822414605, '200': 32.817622519590394, '50': 36.7879988052619}
- TTT by instance size: {'10': 4.28783336161052, '100': 3.8148747056676786, '20': 5.837718742016197, '200': 3.8054704328128754, '50': 3.873919247062104}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 26
- Feedback calls: 26
- Total API calls: 131
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 26
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 21 / 30
- History buckets: 12
- Added candidates: 85
- Duplicate rejections: 0
- History parent selections: 67
- History offspring with history parent: 50
- Final selection from history: False
