# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000084` | valid | 0.514998 | hybrid_sinusoidal_exploit_repair |
| 2 | `cand_000071` | valid | 0.514998 | hybrid_sinusoidal_exploit |
| 3 | `cand_000099` | valid | 0.515015 | hybrid_sinusoidal_de_uniform_refinement |
| 4 | `cand_000102` | valid | 0.515114 | de_sinusoidal_local_search |
| 5 | `cand_000093` | valid | 0.514998 | hybrid_sinusoidal_exploit_repaired |

## Search Best Candidate

- ID: `cand_000084`
- Name: hybrid_sinusoidal_exploit_repair
- Status: valid
- Search Mean AOCC: 0.5149981272998344
- Thought: Combines DE/best/1 exploratory mutation with sinusoidal F modulation and a local Gaussian refinement phase. Population size adapts to budget and dimension. Budget is split: 80% for DE, 20% for local search around the best. All points are clipped to bounds. report_best is called on initial best and all improvements. Seed controls randomness.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.4804370593292344
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.4804370593292344
- Penalized mean AOCC: 0.4804370593292344
- Median AOCC: 0.4673076294287184
- Best AOCC: 0.9934113662844773
- Worst AOCC: 0.08575514657523504
- Mean final error: 1.6571770811003221
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.6755834602269193, 'low_moderate_conditioning': 0.6575730682502001, 'multimodal_strong_global_structure': 0.3268665211011341, 'multimodal_weak_global_structure': 0.17248000264766544, 'separable': 0.6051094462044462}

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
