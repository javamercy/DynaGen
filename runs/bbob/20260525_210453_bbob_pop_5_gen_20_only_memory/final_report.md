# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000060` | valid | 0.532745 | rand_to_best_hybrid |
| 2 | `cand_000091` | valid | 0.52223 | simple_DE |
| 3 | `cand_000086` | valid | 0.522102 | lhs_current_to_pbest_de |
| 4 | `cand_000072` | valid | 0.521569 | de_current_to_best_hybrid_repaired |
| 5 | `cand_000035` | valid | 0.521569 | current_to_best_de |

## Search Best Candidate

- ID: `cand_000060`
- Name: rand_to_best_hybrid
- Status: valid
- Search Mean AOCC: 0.5327453795141803
- Thought: A differential evolution optimizer using DE/rand-to-best/1 mutation to combine exploration from random base vector and exploitation from best solution. Initial population is uniform within bounds. Each generation, for each individual, three distinct random indices are selected (excluding current index). Mutation: pop[r0] + F*(best_x - pop[r0]) + F*(pop[r1] - pop[r2]) with binomial crossover (CR=0.9). Greedy selection maintains elitism. F=0.8, CR=0.9. Seeded to control randomness. Budget is tracked precisely; initial and every improved best are reported via report_best.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5021628046545408
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5021628046545408
- Penalized mean AOCC: 0.5021628046545408
- Median AOCC: 0.5181068555059312
- Best AOCC: 0.9859946574884725
- Worst AOCC: 0.07885589008764934
- Mean final error: 1.0817744537407619
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.7148122537195394, 'low_moderate_conditioning': 0.6718767623066698, 'multimodal_strong_global_structure': 0.3176363999495842, 'multimodal_weak_global_structure': 0.23344335943592806, 'separable': 0.6069880393914084}

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
- History size: 51 / 64
- History buckets: 39
- Added candidates: 94
- Duplicate rejections: 4
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
