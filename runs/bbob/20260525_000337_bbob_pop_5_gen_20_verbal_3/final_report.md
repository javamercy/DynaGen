# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000100` | valid | 0.550025 | cma_es_with_reflection_and_restart |
| 2 | `cand_000077` | valid | 0.545331 | covariance_adaptive_local_refinement |
| 3 | `cand_000085` | valid | 0.54619 | cma_local_refinement |
| 4 | `cand_000038` | valid | 0.539786 | exploration_enhanced_local_refinement |
| 5 | `cand_000071` | valid | 0.539093 | cma_random_refine |

## Search Best Candidate

- ID: `cand_000077`
- Name: covariance_adaptive_local_refinement
- Status: valid
- Search Mean AOCC: 0.5453309343502275
- Thought: This optimizer enhances CMA-ES with a covariance-adaptive local refinement after each restart. After detecting stagnation (20% budget without improvement), it resets the mean with a moderate perturbation, enlarges the step size, resets covariance, and performs a few (up to 5) iterations of a mini CMA-ES (population 2) to quickly adapt to local curvature. The main CMA-ES uses rank-one and active rank-mu covariance update, cumulative step-size adaptation, and budget-adaptive population size. All randomness is controlled by the seed, and points are clipped to bounds.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5377610415347027
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5377610415347027
- Penalized mean AOCC: 0.5377610415347027
- Median AOCC: 0.703757613164569
- Best AOCC: 0.9960304840626691
- Worst AOCC: 0.1001733958315178
- Mean final error: 0.8536783731115434
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.8251346047298072, 'low_moderate_conditioning': 0.8483352889635596, 'multimodal_strong_global_structure': 0.28496220593207056, 'multimodal_weak_global_structure': 0.18234041246218685, 'separable': 0.6101475450716609}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 25
- Feedback calls: 25
- Total API calls: 130
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 3 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 25
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 49 / 64
- History buckets: 39
- Added candidates: 73
- Duplicate rejections: 1
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
