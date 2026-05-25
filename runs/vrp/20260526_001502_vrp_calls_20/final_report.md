# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000079` | valid | 1.82894 | adaptive_regret2_balancing_with_diversified_removal |
| 2 | `cand_000095` | valid | 2.02028 | adaptive_perturbation_regret2 |
| 3 | `cand_000102` | valid | 2.21767 | adaptive_regret2_balancing_enhanced_perturbation |
| 4 | `cand_000103` | valid | 2.23843 | adaptive_perturbation_cycles |
| 5 | `cand_000096` | valid | 2.23843 | best_improvement_tabu_elite_restart |

## Search Best Candidate

- ID: `cand_000079`
- Name: adaptive_regret2_balancing_with_diversified_removal
- Status: valid
- Search Gap: 1.8289388110038747
- Thought: Uses regret-2 construction with squared route distance penalty. Perturbation removes a fraction of customers from both the longest and shortest routes to force load redistribution, with decay over cycles. Repair via regret-2. Local search includes relocate, swap, intra-2opt, cross-2opt, with shake after plateau. Post-construction imbalance reduction move from longest to shortest route. Multi-start with shuffle after 5 non-improving restarts.

## Test Evaluation

- Problem: VRP
- Status: timeout
- Test gap: -0.3693204303749227
- Instances evaluated: 320
- Valid runs: 291 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 29
- Mean gap: -0.3693204303749227
- Penalized mean gap: -0.3693204303749227
- Mean max route distance: 2.717020368905037
- Mean total route distance: 8.824288627666878
- Timeout penalty: 0.0
- Median gap: 0.0066264234627170645
- Worst gap: 5.457914126340885
- Best gap: -58.25068350829989
- Gap by instance size: {'10': -0.121345216977546, '100': 1.656124157976844, '20': -0.7138259764102607, '200': -2.34004923139061, '50': -0.3275058850730407}
- Gap by truck count: {'1': -0.4175855966939034, '3': -0.3275058850730407, '5': 1.656124157976844, '9': -2.34004923139061}
- Error details: timeout on instance_data_200_000: VRP solver timed out after 180.002s (timeout_seconds=180)

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 28
- Feedback calls: 28
- Total API calls: 133
- Failed calls: 1
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 28
- LLM reflection errors: 0

## History

- History enabled: True
- History size: 14 / 64
- History buckets: 12
- Added candidates: 89
- Duplicate rejections: 1
- History parent selections: 54
- History offspring with history parent: 48
- Final selection from history: False
