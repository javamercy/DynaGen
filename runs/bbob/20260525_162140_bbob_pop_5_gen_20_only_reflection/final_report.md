# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000098` | valid | 0.550298 | shade_pbest_reset_adaptive |
| 2 | `cand_000048` | valid | 0.550298 | shade_pbest_dynamic_restart |
| 3 | `cand_000060` | valid | 0.545874 | shade_pbest_local |
| 4 | `cand_000093` | valid | 0.545874 | shade_pbest_dynamic_restart_local_search |
| 5 | `cand_000077` | valid | 0.545874 | shade_pbest_local_refine |

## Search Best Candidate

- ID: `cand_000098`
- Name: shade_pbest_reset_adaptive
- Status: valid
- Search Mean AOCC: 0.5502984635276579
- Thought: Implements SHADE with current-to-pbest/1 mutation, weighted adaptation of F and CR, and diversity-based restart when population standard deviation drops below 1e-6 (relative to bounds range). Removes local refinement to save evaluations for main search. Population size min(10*dim, budget/2). Seed controls randomness. Budget use is strictly tracked.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5543871980121007
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5543871980121007
- Penalized mean AOCC: 0.5543871980121007
- Median AOCC: 0.662084828660159
- Best AOCC: 0.9740747863566303
- Worst AOCC: 0.09457268921288445
- Mean final error: 0.6028084163423713
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.7576459080367036, 'low_moderate_conditioning': 0.7517431738605342, 'multimodal_strong_global_structure': 0.37332404945673325, 'multimodal_weak_global_structure': 0.2978283130706886, 'separable': 0.6308657408055304}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 28
- Feedback calls: 28
- Total API calls: 133
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 28
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
