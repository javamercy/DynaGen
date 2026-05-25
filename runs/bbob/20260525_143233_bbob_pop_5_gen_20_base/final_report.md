# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Mean AOCC | Name |
|---:|---|---|---:|---|
| 1 | `cand_000079` | valid | 0.565315 | simple_de_restart |
| 2 | `cand_000084` | valid | 0.565315 | differential_evolution_restart |
| 3 | `cand_000089` | valid | 0.565315 | simple_de_current_to_best |
| 4 | `cand_000093` | valid | 0.565315 | simple_de_restart |
| 5 | `cand_000073` | valid | 0.567494 | DE_current_to_best_simplified |

## Search Best Candidate

- ID: `cand_000079`
- Name: simple_de_restart
- Status: valid
- Search Mean AOCC: 0.5653145166740001
- Thought: Simplified differential evolution with current-to-best/1 mutation, fixed crossover rate, and restart. Population size scales with dimension but respects budget. Mutation scale F sampled uniformly from [0.5,1.0] per individual. Crossover rate CR is fixed at 0.9. If no improvement after restart_threshold generations, population reinitialized (best retained). Seed controls randomness via np.random.RandomState. Budget tracked strictly.

## Test Evaluation

- Problem: BBOB
- Status: valid
- Test mean AOCC: 0.5474469096704082
- Problem instances evaluated: 48
- Valid runs: 48 / 48
- Scored runs: 48 / 48
- Partial timeout runs: 0
- Mean AOCC: 0.5474469096704082
- Penalized mean AOCC: 0.5474469096704082
- Median AOCC: 0.6149233056248948
- Best AOCC: 0.9960641430220496
- Worst AOCC: 0.07169925880129945
- Mean final error: 0.8861209493654613
- Best final error: 0.0
- AOCC by group: {'high_conditioning_unimodal': 0.75701232811588, 'low_moderate_conditioning': 0.7561061893428586, 'multimodal_strong_global_structure': 0.34914618258853974, 'multimodal_weak_global_structure': 0.2943549072216236, 'separable': 0.6223467970176294}

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
