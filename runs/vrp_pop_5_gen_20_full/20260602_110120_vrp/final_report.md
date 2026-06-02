# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000101` | valid | 2.15458 | aggressive_diversify |
| 2 | `cand_000083` | valid | 2.36827 | dynamic_perturbation_stagnation |
| 3 | `cand_000105` | valid | 2.43748 | diversified_ruin_recreate |
| 4 | `cand_000088` | valid | 2.5298 | adaptive_ruin_recreate |
| 5 | `cand_000104` | valid | 2.59418 | diversity_increased_restarts |

## Search Best Candidate

- ID: `cand_000101`
- Name: aggressive_diversify
- Status: valid
- Search Gap: 2.154581993382788
- Thought: Increased restarts to 50, ruin-recreate removes from all routes (truck_count) with 40% removal per route, deterministic perturbation removes 40% from longest route, both using regret-2 insertion for repair, followed by local search with steepest descent on longest route. This enhances diversification to escape local optima.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: -0.1458727161146897
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: -0.1458727161146897
- Penalized mean gap: -0.1458727161146897
- Mean max route distance: 2.7226349959696314
- Mean total route distance: 8.865289696023403
- Timeout penalty: 0.0
- Median gap: 0.013449569157408067
- Worst gap: 6.492890865675788
- Best gap: -58.11388222240008
- Gap by instance size: {'10': -0.12134521697754573, '100': 1.6836639922988037, '20': -0.5870983818788571, '200': -2.15457916085938, '50': 0.4499951868435306}
- Gap by truck count: {'1': -0.3542217994282014, '3': 0.4499951868435306, '5': 1.6836639922988037, '9': -2.15457916085938}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 35
- Feedback calls: 35
- Total API calls: 140
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 35
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 19 / 20
- History buckets: 13
- Added candidates: 93
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
