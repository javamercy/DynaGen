# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000102` | valid | -0.275828 | balanced_adaptive_ils_v2 |
| 2 | `cand_000083` | valid | -0.142291 | adaptive_ils_enhanced_escape |
| 3 | `cand_000087` | valid | -0.126909 | adaptive_ruin_diversify_maxdist |
| 4 | `cand_000093` | valid | -0.116786 | adaptive_ruin_balance_max |
| 5 | `cand_000104` | valid | -0.0963844 | adaptive_ruin_epsilon_greedy |

## Search Best Candidate

- ID: `cand_000102`
- Name: balanced_adaptive_ils_v2
- Status: valid
- Search Gap: -0.27582801713097066
- Thought: Multi-start ILS with balanced initial solution using regret-3 insertion favoring routes with fewer customers and lower max distance projection; enhanced local search with intra-route 2-opt, inter-route relocate/swap/cross; improved balance procedure that iteratively reduces the longest route via relocation and swap moves, accepting only moves that reduce max distance or reduce total distance when max unchanged; adaptive ruin operators (random, worst, cluster) with simulated annealing and reheating. Tie-breaking: minimize total distance when max distance equal.

## Test Evaluation

- Problem: VRP
- Status: error
- Test gap: inf
- Instances evaluated: 320
- Valid runs: 192 / 320
- Scored runs: 192 / 320
- Partial timeout runs: 0
- Mean gap: -1.954950201816309
- Penalized mean gap: -1.954950201816309
- Mean max route distance: 2.2612311299906005
- Mean total route distance: 12.144863589635738
- Timeout penalty: 0.0
- Median gap: -0.776762637401254
- Worst gap: 3.934787182365718
- Best gap: -59.87942813503395
- Gap by instance size: {'10': None, '100': -0.25375882941721073, '20': None, '200': -5.044132330071236, '50': -0.5669594459604804}
- Gap by truck count: {'1': None, '3': -0.5669594459604804, '5': -0.25375882941721073, '9': -5.044132330071236}
- Error details: error on instance_data_10_000: ValueError: max() iterable argument is empty

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 37
- Feedback calls: 37
- Total API calls: 142
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 37
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 14 / 20
- History buckets: 13
- Added candidates: 93
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
