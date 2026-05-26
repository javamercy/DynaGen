# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000081` | valid | 1.19831 | balance_aware_sa |
| 2 | `cand_000095` | valid | 1.62168 | aggressive_balance_sa |
| 3 | `cand_000100` | valid | 1.6854 | aggressive_balance_sa |
| 4 | `cand_000094` | valid | 1.69726 | adaptive_balance_sa |
| 5 | `cand_000102` | timeout | 1.70117 | balance_enhanced_sa |

## Search Best Candidate

- ID: `cand_000081`
- Name: balance_aware_sa
- Status: valid
- Search Gap: 1.1983130679756893
- Thought: Improved simulated annealing with route-balancing penalty and fewer restarts/iterations to reduce runtime while enhancing solution quality. Construction via Clarke-Wright savings with random shuffle multi-start (5 restarts). Local search uses intra-2opt, inter-relocate, swap, 2-opt* moves targeting the longest route, with a penalty term that discourages moves increasing the maximum route distance disproportionally. Ejection chain perturbation with 1-3 customers from longest route reinserted greedily, but including a balance penalty when evaluating insertions. SA cooling faster (alpha=0.99) with max_iter=2000 per restart to stay within runtime budget.

## Test Evaluation

- Problem: VRP
- Status: invalid
- Test gap: inf
- Instances evaluated: 320
- Valid runs: 250 / 320
- Scored runs: 250 / 320
- Partial timeout runs: 0
- Mean gap: -0.17304574173041665
- Penalized mean gap: -0.17304574173041665
- Mean max route distance: 2.481033830549466
- Mean total route distance: 10.275486279873618
- Timeout penalty: 0.0
- Median gap: 0.2687773302898394
- Worst gap: 7.438031478128092
- Best gap: -58.29767026349082
- Gap by instance size: {'10': 0.9227165467148117, '100': 1.371208175562552, '20': 1.9209920653585473, '200': -3.1218768283254543, '50': 0.051320193922463675}
- Gap by truck count: {'1': 1.129256309192826, '3': 0.051320193922463675, '5': 1.371208175562552, '9': -3.1218768283254543}
- Error details: invalid on instance_data_10_000: VRPSolutionError: VRP routes must visit every customer exactly once
