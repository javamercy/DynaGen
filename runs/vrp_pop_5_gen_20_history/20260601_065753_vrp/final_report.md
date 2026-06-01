# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000104` | valid | 2.35879 | adaptive_memetic |
| 2 | `cand_000097` | valid | 3.52692 | adaptive_memetic_ox_greedy |
| 3 | `cand_000100` | valid | 3.74431 | multi_start_greedy_ls |
| 4 | `cand_000090` | valid | 3.9763 | simplified_memetic |
| 5 | `cand_000088` | valid | 4.50867 | hybrid_memetic_sa_roulette_selection |

## Search Best Candidate

- ID: `cand_000104`
- Name: adaptive_memetic
- Status: valid
- Search Gap: 2.35879223515597
- Thought: Adaptive memetic algorithm with order crossover, swap mutation, and greedy local search. Population size, mutation probability, and local search iterations are adjusted adaptively based on instance size and stagnation. Restart triggers when improvement stalls. Bounded loops by instance size.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: 0.267706529001306
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 0.267706529001306
- Penalized mean gap: 0.267706529001306
- Mean max route distance: 2.7319273031885922
- Mean total route distance: 8.82753747494736
- Timeout penalty: 0.0
- Median gap: 0.03892463433718854
- Worst gap: 7.891516504627303
- Best gap: -57.42579073379002
- Gap by instance size: {'10': -0.12134521697754673, '100': 2.882430345693512, '20': -0.7000901610539471, '200': -2.1675696562477564, '50': 1.4451073335922682}
- Gap by truck count: {'1': -0.41071768901574685, '3': 1.4451073335922682, '5': 2.882430345693512, '9': -2.1675696562477564}

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
- History size: 13 / 20
- History buckets: 11
- Added candidates: 97
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
