# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000161` | timeout | -0.300675 | hybrid_exploitative_alns |
| 2 | `cand_000183` | timeout | -0.0793022 | m4_contract_repair |
| 3 | `cand_000179` | valid | -0.0780593 | hybrid_exploitative_alns_intensified_ls |
| 4 | `cand_000157` | timeout | -0.0504349 | intensified_adaptive_ls |
| 5 | `cand_000176` | timeout | -0.0384863 | explorative_alns |

## Search Best Candidate

- ID: `cand_000161`
- Name: hybrid_exploitative_alns
- Status: timeout
- Search Gap: -0.30067527861853244
- Thought: Combine exploitation-focused ALNS (tight acceptance, high reward for best, local search after repair) with reactive RRT adaptation. Uses farthest-first initialization, three destroy/repair operators with adaptive weights favoring exploitation, greedy local search each iteration, and less frequent restarts.
- Error details: timeout on instances_000: VRP solver timed out after 180.001s (timeout_seconds=180)

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: -1.1800683849195317
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: -1.1800683849195317
- Penalized mean gap: -1.1800683849195317
- Mean max route distance: 2.7000566526511363
- Mean total route distance: 8.641920627197084
- Timeout penalty: 0.0
- Median gap: -0.04265135708335481
- Worst gap: 6.935487104316065
- Best gap: -58.948404494666555
- Gap by instance size: {'10': -0.07311693704044264, '100': -0.3865028082725862, '20': -0.6406727720432248, '200': -4.9174616510619, '50': 0.11741224382049519}
- Gap by truck count: {'1': -0.35689485454183373, '3': 0.11741224382049519, '5': -0.3865028082725862, '9': -4.9174616510619}

## LLM Calls

- Candidate-generation calls: 185
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 185
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 185
- Budget match: True
- LLM reflections enabled: False
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 0
- LLM reflection errors: 0

## History

- History enabled: False
- History size: 0 / 64
- History buckets: 0
- Added candidates: 0
- Duplicate rejections: 0
- History parent selections: 0
- History offspring with history parent: 0
- Final selection from history: False
