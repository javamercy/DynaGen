# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000102` | valid | 0.561016 | hybrid_currenttobest_restart_fixed |
| 2 | `cand_000101` | valid | 0.561016 | hybrid_de_restart |
| 3 | `cand_000075` | valid | 0.561016 | hybrid_currenttobest_restart |
| 4 | `cand_000080` | valid | 0.561016 | simple_de_currenttobest |
| 5 | `cand_000032` | valid | 0.54955 | differential_evolution_currenttobest |

## Search Best Candidate

- ID: `cand_000102`
- Name: hybrid_currenttobest_restart_fixed
- Status: valid
- Search Mean AOCC: 0.5610155522372708
- Thought: Hybrid differential evolution combining current-to-best/1 mutation for fast convergence with stagnation-based restart to escape local optima. Population size adapts to budget (max 4, min 20, budget//5). Uses binomial crossover with CR=0.9 and F=0.8. Restarts when no best improvement for ceil(budget/10) evaluations. For very small budgets (<4), falls back to random sampling. Seed controls randomness. Adheres to interface and safety rules.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5084041427082482
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5084041427082482
- Penalized mean AOCC: 0.5084041427082482
- Median AOCC: 0.5537585937117913
- Best AOCC: 0.9920893091528169
- Worst AOCC: 0.07973277706281294
- Mean final error: 1.306452638840233
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.7364112222227238, 'low_moderate_conditioning': 0.6894151758805654, 'multimodal_strong_global_structure': 0.318137969121787, 'multimodal_weak_global_structure': 0.23244911863724713, 'separable': 0.6018094343133812}

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
- History size: 45 / 64
- History buckets: 38
- Added candidates: 96
- Duplicate rejections: 6
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
