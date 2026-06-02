# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000083` | valid | 2.21088 | load_balanced_shake_enhanced |
| 2 | `cand_000086` | timeout | 2.25942 | diversified_shake_regret_reinsert |
| 3 | `cand_000072` | valid | 3.02773 | gentle_shake_cheapest_reinsert |
| 4 | `cand_000076` | valid | 3.03874 | GentleShakeVRP |
| 5 | `cand_000056` | valid | 3.05218 | shaken_regret_balancer |

## Search Best Candidate

- ID: `cand_000083`
- Name: load_balanced_shake_enhanced
- Status: valid
- Search Gap: 2.21087965407102
- Thought: Hybrid algorithm: builds routes using regret-2/3 insertion where tie-breaking among equal-cost insertions favors routes with lower current total distance and then lower max route distance. After construction, applies bounded 2-opt, relocate, and swap improvements. Then for increased shake intensity (remove 20% of customers from longest route per shake) and more shake iterations (20% of n+2), performs perturbation and reinsertion using regret-2 insertion to balance loads, followed by local search. Reports incumbents via report_best_vrp. Returns best solution found.

## Test Evaluation

- Problem: VRP
- Status: valid
- Test gap: 0.3954173841509622
- Instances evaluated: 320
- Valid runs: 320 / 320
- Scored runs: 320 / 320
- Partial timeout runs: 0
- Mean gap: 0.3954173841509622
- Penalized mean gap: 0.3954173841509622
- Mean max route distance: 2.7345549585157802
- Mean total route distance: 8.935429928841183
- Timeout penalty: 0.0
- Median gap: 0.0546301453076974
- Worst gap: 10.049731427571636
- Best gap: -56.1527491975489
- Gap by instance size: {'10': 0.23402071616404507, '100': 2.980928999402415, '20': -0.4351263584083909, '200': -0.5106994940430707, '50': -0.2920369423601872}
- Gap by truck count: {'1': -0.10055282112217293, '3': -0.2920369423601872, '5': 2.980928999402415, '9': -0.5106994940430707}

## LLM Calls

- Candidate-generation calls: 105
- Reflection calls: 27
- Feedback calls: 27
- Total API calls: 132
- Failed calls: 0
- Main LLM model: deepseek-v4-flash
- Feedback LLM model: deepseek-v4-flash
- Configured candidate-generation budget: 105
- Budget match: True
- LLM reflections enabled: True
- LLM reflection cadence: every 2 generations
- LLM reflection model: deepseek-v4-flash
- LLM reflections: 26
- LLM reflection errors: 1

## History

- History enabled: False
- History size: 0 / 20
- History buckets: 0
- Added candidates: 0
- Duplicate rejections: 0
- History parent selections: 0
- History offspring with history parent: 0
- Final selection from history: False
