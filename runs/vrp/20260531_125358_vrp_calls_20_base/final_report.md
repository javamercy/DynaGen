# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000173` | timeout | 2.81154 | simple_ga_regret2 |
| 2 | `cand_000184` | timeout | 3.53121 | intensified_memetic_ga |
| 3 | `cand_000177` | timeout | 3.53333 | iterated_local_search_vrp |
| 4 | `cand_000185` | timeout | 3.62189 | diversify_ga_restart |
| 5 | `cand_000181` | timeout | 3.75153 | adaptive_ruin_ga_aggressive |

## Search Best Candidate

- ID: `cand_000173`
- Name: simple_ga_regret2
- Status: timeout
- Search Gap: 2.811542593239703
- Thought: Simplified GA using regret-2 construction for initial population, order crossover (OX1), swap mutation, tournament selection, elitism, and intra-route 2-opt improvement. Objective: minimize max route distance then total distance. Bounded loops, deterministic tie-breaking.
- Error details: timeout on instances_000: VRP solver timed out after 180s (timeout_seconds=180)

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: 0.25873674038628897
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 0.25873674038628897
- Penalized mean gap: 0.25873674038628897
- Mean max route distance: 2.7307682666368813
- Mean total route distance: 8.800125205477517
- Timeout penalty: 0.0
- Median gap: 0.05225089613431895
- Worst gap: 7.5013129348227565
- Best gap: -56.92576402395128
- Gap by instance size: {'10': -0.12134521697755085, '100': 3.139755801259207, '20': -0.7138259764102632, '200': -1.3600766871307883, '50': 0.34917578119084014}
- Gap by truck count: {'1': -0.41758559669390705, '3': 0.34917578119084014, '5': 3.139755801259207, '9': -1.3600766871307883}

## LLM Calls

- Candidate-generation calls: 185
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 185
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 185
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
