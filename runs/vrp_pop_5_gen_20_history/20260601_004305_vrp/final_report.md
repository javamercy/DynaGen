# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000103` | timeout | 6.69448 | SimpleALNS_WorstRemoval |
| 2 | `cand_000036` | timeout | 6.80953 | AdaptiveLargeNeighborhoodSearch |
| 3 | `cand_000091` | timeout | 7.01 | hybrid_ga_ls |
| 4 | `cand_000104` | timeout | 8.65317 | Adaptive_GRASP_VND |
| 5 | `cand_000086` | timeout | 10.1318 | GRASP_with_VND |

## Search Best Candidate

- ID: `cand_000103`
- Name: SimpleALNS_WorstRemoval
- Status: timeout
- Search Gap: 6.694484634240872
- Thought: Replace random removal with worst removal in SimpleALNS. Keep nearest-neighbor TSP + DP split initialization, greedy insertion repair, simulated annealing acceptance, and deterministic tie-breaking. Worst removal picks customers from routes with larger distances, helping improve max route distance.
- Error details: timeout on instances_000: VRP solver timed out after 200.001s (timeout_seconds=200)

## Test Evaluation

- Problem: VRP
- Status: error
- Test gap: inf
- Instances evaluated: 320
- Valid runs: 192 / 320
- Scored runs: 192 / 320
- Partial timeout runs: 0
- Mean gap: 5.589847909173169
- Penalized mean gap: 5.589847909173169
- Mean max route distance: 2.428568848248163
- Mean total route distance: 13.289908334235108
- Timeout penalty: 0.0
- Median gap: 5.515376897449576
- Worst gap: 19.55791275015159
- Best gap: -51.832180467306465
- Gap by instance size: {'10': None, '100': 8.24870291333151, '20': None, '200': 4.251383631790769, '50': 4.26945718239723}
- Gap by truck count: {'1': None, '3': 4.26945718239723, '5': 8.24870291333151, '9': 4.251383631790769}
- Error details: error on instance_data_10_000: TypeError: 'numpy.float64' object is not iterable

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 105
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: False
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 0
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 20 / 20
- History buckets: 13
- Added candidates: 96
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
