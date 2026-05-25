# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000079` | valid | 0.81763 | ProbabilisticRegretMultiSwap |
| 2 | `cand_000105` | valid | 0.975163 | BalancedRegretThreshold_StrongPerturb |
| 3 | `cand_000084` | valid | 1.07607 | Hybrid SALoadBalance |
| 4 | `cand_000096` | valid | 1.32566 | AdaptiveLoadBalanceThreshold |
| 5 | `cand_000099` | valid | 1.3305 | EnhancedDeterministicRegretThreshold_AltRestart |

## Search Best Candidate

- ID: `cand_000079`
- Name: ProbabilisticRegretMultiSwap
- Status: valid
- Search Gap: 0.8176300971177397
- Thought: Probabilistic regret-based construction with random tie-breaking among top 3 regrets to increase diversity; threshold-accepting local search with 2-opt, relocate, swap, cross-exchange; periodic worst-customer relocation every 30 iterations; when stagnation detected (30 iterations without improvement), multi-swap perturbation (swap two random customers from different routes) if accepted under threshold; restart with reversed tie-breaking after 100 iterations without improvement.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: -0.07568329428324347
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: -0.07568329428324347
- Penalized mean gap: -0.07568329428324347
- Mean max route distance: 2.729040486159195
- Mean total route distance: 8.856643157585616
- Timeout penalty: 0.0
- Median gap: 0.02730690725417305
- Worst gap: 18.662854597733617
- Best gap: -57.59859700240889
- Gap by instance size: {'10': 0.3700574339736767, '100': 1.079246491173834, '20': 0.7812510941097003, '200': -2.3245268509235077, '50': -0.2844446397499207}
- Gap by truck count: {'1': 0.5756542640416885, '3': -0.2844446397499207, '5': 1.079246491173834, '9': -2.3245268509235077}

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
- History size: 15 / 64
- History buckets: 12
- Added candidates: 94
- Duplicate rejections: 0
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
