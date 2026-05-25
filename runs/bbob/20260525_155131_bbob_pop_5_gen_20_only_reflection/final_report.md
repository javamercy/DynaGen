# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000103` | valid | 0.503635 | Adaptive Cauchy Restart DE |
| 2 | `cand_000089` | valid | 0.50376 | focused_restart_de |
| 3 | `cand_000097` | valid | 0.50376 | focused_restart_de_success_cauchy |
| 4 | `cand_000101` | valid | 0.501694 | adaptive_restart_de_cauchy_improved |
| 5 | `cand_000073` | valid | 0.501694 | cauchy_de_restart_adapt |

## Search Best Candidate

- ID: `cand_000103`
- Name: Adaptive Cauchy Restart DE
- Status: valid
- Search Mean AOCC: 0.503634828034518
- Thought: DE/best/1 with dither F=0.5+0.5*rand, CR=0.9, stagnation restart limit max(5,NP//2,dim). Up to 3 restarts: reinitialize with 50% uniform and 50% Cauchy around best (scale=0.1*(ub-lb)). After restart, focused local search with 90% Gaussian (initial step=0.01*(ub-lb), decay 0.9 each iteration) and 10% Cauchy with success-based adaptation: base scale=0.05*(ub-lb), increase by factor 1.05 on improvement, decrease by 0.95 on failure. Seed controls all randomness.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.47723233897418016
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.47723233897418016
- Penalized mean AOCC: 0.47723233897418016
- Median AOCC: 0.48663686947802187
- Best AOCC: 0.9940664539571955
- Worst AOCC: 0.07762975348202582
- Mean final error: 1.1664505816256059
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.6826106491580333, 'low_moderate_conditioning': 0.665254432551993, 'multimodal_strong_global_structure': 0.3217867079432073, 'multimodal_weak_global_structure': 0.1655766587589827, 'separable': 0.588537665174247}

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
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 24
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
