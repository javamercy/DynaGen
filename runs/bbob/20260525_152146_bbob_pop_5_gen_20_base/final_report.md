# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000099` | valid | 0.560226 | binomial_crossover_de_rand_to_best_restart |
| 2 | `cand_000093` | valid | 0.556854 | exponential_diversify_de_rand_to_best_restart |
| 3 | `cand_000086` | valid | 0.556854 | exponential_diversify_de_rand_to_best_restart |
| 4 | `cand_000088` | valid | 0.548961 | e3_hybrid_recombination |
| 5 | `cand_000083` | valid | 0.548961 | diversify_de_rand_to_best_restart |

## Search Best Candidate

- ID: `cand_000099`
- Name: binomial_crossover_de_rand_to_best_restart
- Status: valid
- Search Mean AOCC: 0.5602259732656504
- Thought: This optimizer modifies the parent by replacing exponential crossover with binomial crossover, which is the classic DE crossover. Mutation is DE/rand-to-best/1 with random F per individual. Restart mechanism preserved. Population size min(25, max(5, budget//2)). Stagnation limit max(1, budget//(4*pop_size)). Seed controls all randomness. Budget tracked precisely; report_best called on initial best and every improvement.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5742966610020701
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5742966610020701
- Penalized mean AOCC: 0.5742966610020701
- Median AOCC: 0.709912007211528
- Best AOCC: 0.9903510469817455
- Worst AOCC: 0.08629041649640848
- Mean final error: 0.9352244695627904
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.7776257622844991, 'low_moderate_conditioning': 0.734276510453188, 'multimodal_strong_global_structure': 0.3212934039010485, 'multimodal_weak_global_structure': 0.3790987619282019, 'separable': 0.6911848363336366}

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

- History enabled: False
- History size: 0 / 64
- History buckets: 0
- Added candidates: 0
- Duplicate rejections: 0
- History parent selections: 0
- History offspring with history parent: 0
- Final selection from history: False
