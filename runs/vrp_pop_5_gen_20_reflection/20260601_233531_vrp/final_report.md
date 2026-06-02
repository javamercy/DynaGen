# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000056` | valid | 0.564044 | penalty_balancing_2optstar |
| 2 | `cand_000093` | valid | 0.566792 | reduced_penalty |
| 3 | `cand_000097` | valid | 0.641151 | dynamic_balance_045 |
| 4 | `cand_000065` | valid | 0.724584 | penalty_increased |
| 5 | `cand_000096` | valid | 0.770679 | strong_penalty_regret |

## Search Best Candidate

- ID: `cand_000056`
- Name: penalty_balancing_2optstar
- Status: valid
- Search Gap: 0.5640436455538925
- Thought: Multi-start construction with adaptive alpha merges initial routes, then local search (2-opt intra, relocate/swap inter, plus 2-opt* inter). Perturbation uses ruin-and-recreate: remove customers with probability proportional to route max-distance contribution, then reinsert using regret-2 heuristic with a penalty for inserting into long routes to promote balancing. Five perturbation cycles with decreasing removal ratio (30% to 10%). Final improvement after all cycles. Tie-breaking favors lower max then lower total distance.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: -0.010473740544679599
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: -0.010473740544679599
- Penalized mean gap: -0.010473740544679599
- Mean max route distance: 2.728217186361611
- Mean total route distance: 8.730338516149128
- Timeout penalty: 0.0
- Median gap: 0.021798991070118706
- Worst gap: 7.7402913263879976
- Best gap: -58.28733793732776
- Gap by instance size: {'10': -0.08483044067336194, '100': 2.1570989060118593, '20': 0.01896867620157106, '200': -3.3931216674142637, '50': 1.2495158231507975}
- Gap by truck count: {'1': -0.032930882235895444, '3': 1.2495158231507975, '5': 2.1570989060118593, '9': -3.3931216674142637}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 33
- Feedback calls: 33
- Total API calls: 138
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 33
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
