# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000079` | valid | 0.603225 | cma_es_local_refine |
| 2 | `cand_000092` | valid | 0.599809 | cma_es_restarts_double_pop_fixed_local |
| 3 | `cand_000059` | valid | 0.597835 | cma_es_restart_condition_restart |
| 4 | `cand_000094` | valid | 0.596258 | cma_mirror_local |
| 5 | `cand_000058` | valid | 0.595363 | cma_es_restart_condition_number |

## Search Best Candidate

- ID: `cand_000079`
- Name: cma_es_local_refine
- Status: valid
- Search Mean AOCC: 0.603225266001353
- Thought: CMA-ES with restarts, population doubling, and local refinement from the best point after each restart. Uses condition number trigger (1e14) and stagnation check (max_no_improve = 50 + 0.2*d). After each restart, if budget remains, runs a brief local search: sample 10 candidate points as best + sigma_small * randn, clip to bounds, evaluate, update best. This improves multimodal and separable performance by fine-tuning near good solutions. Standard CMA-ES update with rank-1/rank-mu, cumulative step-size adaptation, and resampling for bounds. Seed controls all randomness.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5939857386316797
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5939857386316797
- Penalized mean AOCC: 0.5939857386316797
- Median AOCC: 0.7272533035519767
- Best AOCC: 0.9681194347072917
- Worst AOCC: 0.09019276892608645
- Mean final error: 0.6955578337779897
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.8316028294863906, 'low_moderate_conditioning': 0.8360997961175363, 'multimodal_strong_global_structure': 0.43075145045242236, 'multimodal_weak_global_structure': 0.3201104058921298, 'separable': 0.5997870227070906}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 23
- Feedback calls: 23
- Total API calls: 128
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 23
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
