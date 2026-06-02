# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000079` | timeout | 0.52199 | longest_route_biased_ruin |
| 2 | `cand_000075` | timeout | 1.55903 | simpler_ruin_best_improve_ls |
| 3 | `cand_000103` | valid | 1.81712 | longest_route_focused_sa |
| 4 | `cand_000077` | timeout | 1.96674 | biased_ruin_minimax_ls |
| 5 | `cand_000087` | timeout | 2.09637 | balanced_ruin_longest_recreate |

## Search Best Candidate

- ID: `cand_000079`
- Name: longest_route_biased_ruin
- Status: timeout
- Search Gap: 0.5219897738278229
- Thought: Start with minimax construction. Iteratively ruin-and-recreate where ruin removal is biased towards customers from the longest route (probability proportional to route distance). Reconstruct using minimax insertion with tie-breaking: minimize new max route distance, then new route total distance, then prefer route with smallest current distance to balance loads. Apply inter-route relocate and swap (first improvement reducing max distance) and intra-route 2-opt, with extra focus on the longest route. Accept via simulated annealing. Keep best solution.
- Error details: invalid on instances_000: VRPSolutionError: VRP routes must visit every customer exactly once

## Test Evaluation

- Problem: VRP
- Status: invalid
- Test gap: inf
- Instances evaluated: 320
- Valid runs: 153 / 320
- Scored runs: 153 / 320
- Partial timeout runs: 0
- Mean gap: 0.03870800581984694
- Penalized mean gap: 0.03870800581984694
- Mean max route distance: 3.209680055263949
- Mean total route distance: 4.343230953724794
- Timeout penalty: 0.0
- Median gap: -0.0016050992269315367
- Worst gap: 6.19024930366659
- Best gap: -4.197078718734086
- Gap by instance size: {'10': 0.10874376938212753, '100': 1.8335943028809234, '20': -0.1923062191643799, '200': 0.817911681841919, '50': 0.08234496774729337}
- Gap by truck count: {'1': -0.04178122489112618, '3': 0.08234496774729337, '5': 1.8335943028809234, '9': 0.817911681841919}
- Error details: invalid on instance_data_50_000: VRPSolutionError: VRP routes must visit every customer exactly once

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 26
- Feedback calls: 26
- Total API calls: 131
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 26
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
