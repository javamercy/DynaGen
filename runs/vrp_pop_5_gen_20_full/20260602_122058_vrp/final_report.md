# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000041` | valid | 1.66014 | reactive_tabu_critical_focus |
| 2 | `cand_000096` | timeout | 2.2633 | adaptive_tabu_with_fast_perturb |
| 3 | `cand_000076` | valid | 2.29142 | reactive_tabu_with_fast_perturb |
| 4 | `cand_000093` | valid | 2.59165 | balanced_restart_v1 |
| 5 | `cand_000095` | valid | 2.59165 | regret_restart_accept_v4 |

## Search Best Candidate

- ID: `cand_000041`
- Name: reactive_tabu_critical_focus
- Status: valid
- Search Gap: 1.6601380454847798
- Thought: Combines reactive tabu search with critical route intensification and load-balancing perturbation. Uses farthest-first seeds and greedy min-max insertion, then tabu search with relocate, swap, 2-opt, cross-2-opt* neighborhoods. Tabu tenure adapts based on improvement cycles. When stuck, perturbs by removing customers from the longest route(s) and reinserts via regret-2, then applies extra 2-opt on the longest route. Restarts from scratch with a different seed after extended stagnation.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: 1.3594831678400416
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 1.3594831678400416
- Penalized mean gap: 1.3594831678400416
- Mean max route distance: 2.7679014767346195
- Mean total route distance: 8.914898772438455
- Timeout penalty: 0.0
- Median gap: 1.3867843537320976
- Worst gap: 20.646373613767974
- Best gap: -58.67995523771621
- Gap by instance size: {'10': 4.6559338120568645, '100': 1.902366640763106, '20': 1.1081299980245394, '200': -2.413732698335748, '50': 1.5447180866914467}
- Gap by truck count: {'1': 2.882031905040702, '3': 1.5447180866914467, '5': 1.902366640763106, '9': -2.413732698335748}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 36
- Feedback calls: 36
- Total API calls: 141
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 36
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 18 / 20
- History buckets: 14
- Added candidates: 90
- Duplicate rejections: 0
- History parent selections: 57
- History offspring with history parent: 44
- Final selection from history: False
