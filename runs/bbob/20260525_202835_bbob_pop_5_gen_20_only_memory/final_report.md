# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000050` | valid | 0.562498 | diversified_de_restart |
| 2 | `cand_000102` | valid | 0.562498 | differential_evolution_restart |
| 3 | `cand_000101` | valid | 0.563341 | hybrid_de_nm |
| 4 | `cand_000082` | valid | 0.550606 | de_lhs_init |
| 5 | `cand_000085` | valid | 0.546079 | exponential_crossover_de_restart |

## Search Best Candidate

- ID: `cand_000050`
- Name: diversified_de_restart
- Status: valid
- Search Mean AOCC: 0.5624976254443927
- Thought: Differential evolution with larger population, dithering F and CR for exploration, and restart on stagnation. Initializes population uniformly. Mutation uses current-to-best/1 with dithering F and CR per generation to vary search trajectories. If no improvement after a fraction of budget, reinitialize a random subset of the population to escape local optima. Seed controls randomness. Budget is strictly respected with early breaks. Reports best upon initialization and each improvement.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5130339779497279
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5130339779497279
- Penalized mean AOCC: 0.5130339779497279
- Median AOCC: 0.528171182532059
- Best AOCC: 0.9805866046298335
- Worst AOCC: 0.09592776911317251
- Mean final error: 0.9950371187099517
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.5504599174911976, 'low_moderate_conditioning': 0.7190782626862958, 'multimodal_strong_global_structure': 0.32391043918177387, 'multimodal_weak_global_structure': 0.3930894016277307, 'separable': 0.6198407257089553}

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
- History size: 52 / 64
- History buckets: 39
- Added candidates: 97
- Duplicate rejections: 3
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
