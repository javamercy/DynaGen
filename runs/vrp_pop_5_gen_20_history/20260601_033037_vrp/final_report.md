# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000104` | timeout | 1.29456 | reactive_tabu_search_vrp_variant |
| 2 | `cand_000102` | valid | 4.71776 | hybrid_tabu_ruin_recreate |
| 3 | `cand_000071` | valid | 5.53339 | tabu_search_vrp |
| 4 | `cand_000084` | valid | 5.53339 | adaptive_tabu_search_vrp |
| 5 | `cand_000089` | timeout | 6.30467 | reactive_tabu_search_vrp |

## Search Best Candidate

- ID: `cand_000104`
- Name: reactive_tabu_search_vrp_variant
- Status: timeout
- Search Gap: 1.2945576916610062
- Thought: Modified reactive tabu search with corrected tenure scheduling: tenure increases on non-improving moves (stagnation) and decreases only when a move improves the best solution. Added streak-based adjustments: if improvement streak > 3, decrement tenure by 2; if stagnation streak > 3, increment tenure by 3. Stagnation limit increased to max(50, n*3) for larger instances. Construction uses min-max insertion. Tie-breaking by total route distance.
- Error details: timeout on instances_000: VRP solver timed out after 300.001s (timeout_seconds=300)

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: 1.7455648495817635
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 1.7455648495817635
- Penalized mean gap: 1.7455648495817635
- Mean max route distance: 2.7815947691151943
- Mean total route distance: 8.978466945653938
- Timeout penalty: 0.0
- Median gap: 0.8262544893813895
- Worst gap: 22.239003631307355
- Best gap: -58.808547470530016
- Gap by instance size: {'10': 0.8560545388658931, '100': 3.737986735548507, '20': 4.115116631282909, '200': -2.2853577049851332, '50': 2.3040240471966413}
- Gap by truck count: {'1': 2.4855855850744013, '3': 2.3040240471966413, '5': 3.737986735548507, '9': -2.2853577049851332}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 0
- Feedback calls: 0
- Total API calls: 105
- Failed calls: 1
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
- History size: 19 / 20
- History buckets: 13
- Added candidates: 96
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
