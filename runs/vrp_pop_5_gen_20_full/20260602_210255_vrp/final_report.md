# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000083` | timeout | 3.44403 | adaptive_diverse_search |
| 2 | `cand_000104` | valid | 3.80881 | diversification_shake_escape |
| 3 | `cand_000084` | valid | 3.82587 | balanced_adaptive_perturb |
| 4 | `cand_000099` | valid | 4.18468 | adaptive_escape_schedule |
| 5 | `cand_000082` | valid | 4.51555 | adaptive_farthest_perturb |

## Search Best Candidate

- ID: `cand_000083`
- Name: adaptive_diverse_search
- Status: timeout
- Search Gap: 3.4440270669709037
- Thought: Improved upon parent adaptive_perturb_stagnation by increasing perturbation diversity (random mix of relocate, swap, 2-opt, cross) and intensifying local search with VNS-style shaking after stagnation. Also increased max_iter to min(n*30,2000) and max_restarts to min(30,n/4) for broader exploration.
- Error details: timeout on instances_000: VRP solver timed out after 300.001s (timeout_seconds=300)

## Test Evaluation

- Problem: VRP
- Status: error
- Test gap: inf
- Instances evaluated: 320
- Valid runs: 200 / 320
- Scored runs: 200 / 320
- Partial timeout runs: 0
- Mean gap: 2.2035740366710614
- Penalized mean gap: 2.2035740366710614
- Mean max route distance: 2.3743091712858155
- Mean total route distance: 12.449896729950538
- Timeout penalty: 0.0
- Median gap: 2.6063135965006756
- Worst gap: 15.636487475081294
- Best gap: -56.61939359873621
- Gap by instance size: {'10': 0.0047553046149254085, '100': 4.645805218492745, '20': None, '200': -0.1819973687067934, '50': 2.42176660173425}
- Gap by truck count: {'1': 0.0047553046149254085, '3': 2.42176660173425, '5': 4.645805218492745, '9': -0.1819973687067934}
- Error details: error on instance_data_10_002: IndexError: Cannot choose from an empty sequence

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 29
- Feedback calls: 29
- Total API calls: 134
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 29
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 18 / 20
- History buckets: 13
- Added candidates: 81
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
