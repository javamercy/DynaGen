# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000089` | valid | 0.530128 | covariance_adaptive_de_restart |
| 2 | `cand_000094` | valid | 0.527917 | diverse_adaptive_de_with_crowding |
| 3 | `cand_000083` | valid | 0.525561 | intensified_local_search_de |
| 4 | `cand_000060` | valid | 0.523367 | adaptive_current_to_best_with_restart |
| 5 | `cand_000100` | valid | 0.521159 | adaptive_de_cov_restart |

## Search Best Candidate

- ID: `cand_000089`
- Name: covariance_adaptive_de_restart
- Status: valid
- Search Mean AOCC: 0.5301275819251798
- Thought: Adaptive DE with current-to-best/1 and rand/1 strategies selected via exponential smoothing of success rates, greedy selection, and localized restart using covariance matrix estimated from recent successful steps. The restart spread is scaled inversely with sqrt(dimension) for better high-dimensional exploration. Seed controls randomness.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5072266198642452
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5072266198642452
- Penalized mean AOCC: 0.5072266198642452
- Median AOCC: 0.5801514836476404
- Best AOCC: 0.9918391505066716
- Worst AOCC: 0.07669576392640282
- Mean final error: 1.1749422907323408
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.7096079283832958, 'low_moderate_conditioning': 0.637932386120917, 'multimodal_strong_global_structure': 0.3236770367175049, 'multimodal_weak_global_structure': 0.27139684364322736, 'separable': 0.6196600577076156}

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
- History size: 52 / 64
- History buckets: 38
- Added candidates: 96
- Duplicate rejections: 0
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
