# DynaGen Final Report

## Final Population

| Rank | Candidate     | Status | Search Distance | Name                    |
|-----:|---------------|--------|----------------:|-------------------------|
|    1 | `cand_000010` | valid  |         11.4529 | m6_swap_perturb         |
|    2 | `cand_000017` | valid  |         11.9673 | adaptive_regret_perturb |
|    3 | `cand_000018` | valid  |         13.5679 | m4_contract_repair      |

## Search Best Candidate

- ID: `cand_000010`
- Name: m6_swap_perturb
- Status: valid
- Search Distance: 11.45288823534332
- Thought: Modified m6_diversify_search: replaced relocation perturbation (move customers from longest route) with
  random inter-route swap perturbation (1-3 swaps) while preserving stochastic regret construction and deterministic
  local search.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test distance: 5.537658864308722
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 5.537658864308722
- Penalized mean gap: 5.537658864308722
- Mean max route distance: 2.851544661097156
- Mean total route distance: 9.639965750670838
- Timeout penalty: 0.0
- Median gap: 4.21928620508348
- Worst gap: 27.997596949178046
- Best gap: -50.47584974155735
- Score by instance size: {'10': -0.019724468521927208, '100': 12.64661094926188, '20': -0.041077779559690886, '200':
  9.650670826325733, '50': 5.451814794037613}
- Score by truck count: {'1': -0.030401124040809045, '3': 5.451814794037613, '5': 12.64661094926188, '9':
  9.650670826325733}

## LLM Calls

- Candidate-generation calls: 18
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 18
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 18
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
