# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000151` | valid | 0.61791 | jade_pattern_cycle |
| 2 | `cand_000139` | valid | 0.617698 | jade_cma_local |
| 3 | `cand_000111` | valid | 0.615316 | jade_local_refinement |
| 4 | `cand_000152` | valid | 0.615521 | AdaptiveLocalJADE_RestrictedLocal |
| 5 | `cand_000124` | valid | 0.612679 | jade_cma_local_refinement |

## Search Best Candidate

- ID: `cand_000151`
- Name: jade_pattern_cycle
- Status: valid
- Search Mean AOCC: 0.6179097638787112
- Thought: JADE optimizer with budget-adaptive population size, pbest ratio, and scaling factor. Uses success-based parameter adaptation (Lehmer mean for F, arithmetic mean for CR), finite archive, diversity-aware restart, and pattern search refinement. Local refinement cycles through all coordinates with per-coordinate step size adaptation, using at most 5% of remaining evaluations. Restart stagnation threshold scales with remaining budget. Rand/1 mutation is more aggressive in low diversity (70% probability). Seed controls all randomness.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.6313477108585944
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.6313477108585944
- Penalized mean AOCC: 0.6313477108585944
- Median AOCC: 0.7692729359297599
- Best AOCC: 0.9927573046582269
- Worst AOCC: 0.08947536641855246
- Mean final error: 0.6524066977490123
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.7947855768439298, 'low_moderate_conditioning': 0.8152964115657917, 'multimodal_strong_global_structure': 0.27515590591569394, 'multimodal_weak_global_structure': 0.4525231941962338, 'separable': 0.8557672059127622}

## LLM Calls

- Candidate-generation calls: 155
- Reflection calls: 50
- Feedback calls: 50
- Total API calls: 205
- Failed calls: 1
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 155
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 50
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 56 / 64
- History buckets: 39
- Added candidates: 143
- Duplicate rejections: 2
- History parent selections: 78
- History offspring with history parent: 69
- Final selection from history: False
