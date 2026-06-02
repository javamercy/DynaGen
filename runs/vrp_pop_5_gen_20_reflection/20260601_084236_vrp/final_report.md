# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000064` | timeout | 2.27527 | load_balanced_reduction |
| 2 | `cand_000103` | valid | 2.41747 | balanced_local_search_improved |
| 3 | `cand_000066` | timeout | 2.61908 | load_balanced_reduction |
| 4 | `cand_000072` | timeout | 2.6372 | fast_balanced_reduction |
| 5 | `cand_000087` | valid | 2.64727 | balanced_searcher |

## Search Best Candidate

- ID: `cand_000064`
- Name: load_balanced_reduction
- Status: timeout
- Search Gap: 2.2752699986429126
- Thought: Enhances aggressive_reduction by incorporating load balancing: accepts max-reduction moves that also reduce total distance when max unchanged, and adds perturbation that swaps customers between longest and shortest routes. Increases diversity to escape plateaus.
- Error details: timeout on instances_000: VRP solver timed out after 300s (timeout_seconds=300)

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: 1.4310455017095551
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 1.4310455017095551
- Penalized mean gap: 1.4310455017095551
- Mean max route distance: 2.773444313050269
- Mean total route distance: 8.89531054024467
- Timeout penalty: 0.0
- Median gap: 1.3718148212251964
- Worst gap: 24.211498426537112
- Best gap: -58.012875750428464
- Gap by instance size: {'10': 3.0144380216916318, '100': 2.9515069552844304, '20': 2.7381016158346094, '200': -2.518650633723085, '50': 0.9698315494601889}
- Gap by truck count: {'1': 2.8762698187631206, '3': 0.9698315494601889, '5': 2.9515069552844304, '9': -2.518650633723085}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 32
- Feedback calls: 32
- Total API calls: 137
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 32
- LLM reflection errors: 0

## History

- History enabled: False
- History size: 0 / 20
- History buckets: 0
- Added candidates: 0
- Duplicate rejections: 0
- History parent selections: 0
- History offspring with history parent: 0
- Final selection from history: False
