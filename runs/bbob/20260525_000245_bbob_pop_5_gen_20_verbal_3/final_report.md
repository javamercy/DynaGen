# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000065` | valid | 0.568003 | cma_es_with_local_refinement |
| 2 | `cand_000102` | valid | 0.560874 | cma_es_restart_diverse |
| 3 | `cand_000085` | valid | 0.560874 | cma_es_adaptive_restart |
| 4 | `cand_000079` | valid | 0.560874 | cma_es_restart_diverse |
| 5 | `cand_000078` | valid | 0.560874 | adaptive_cma_es_diverse_restart |

## Search Best Candidate

- ID: `cand_000065`
- Name: cma_es_with_local_refinement
- Status: valid
- Search Mean AOCC: 0.568003363812175
- Thought: CMA-ES with rank-one/rank-mu update, cumulative step size adaptation, aggressive restart, and local refinement phase. Initializes at random feasible point. Population size adapts to remaining budget. Restart triggers on stagnation (no improvement, low fitness diversity, or small sigma). When no improvement persists, performs a few local perturbations around the current best with step size decaying over budget. All points clipped to bounds. Seed controls all randomness.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5682211372783149
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5682211372783149
- Penalized mean AOCC: 0.5682211372783149
- Median AOCC: 0.6695186911920852
- Best AOCC: 0.9944134411969808
- Worst AOCC: 0.08709367254829135
- Mean final error: 1.0007165251154746
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.800812098142725, 'low_moderate_conditioning': 0.8008889385448275, 'multimodal_strong_global_structure': 0.38358831931243065, 'multimodal_weak_global_structure': 0.3046733067259725, 'separable': 0.5976765839189215}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 24
- Feedback calls: 24
- Total API calls: 129
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 24
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 50 / 64
- History buckets: 39
- Added candidates: 76
- Duplicate rejections: 1
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
