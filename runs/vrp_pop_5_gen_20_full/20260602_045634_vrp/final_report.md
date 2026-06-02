# DynaGen Final Report

## Final Population

| Rank | Candidate | Status | Search Gap | Name |
|---:|---|---|---:|---|
| 1 | `cand_000063` | valid | 5.11034 | high_cost_shake |
| 2 | `cand_000037` | valid | 5.26824 | adaptive_ils_vrp |
| 3 | `cand_000078` | valid | 5.34424 | randomized_longest_route_elimination |
| 4 | `cand_000072` | valid | 5.44849 | mixed_shake_deterministic |
| 5 | `cand_000061` | valid | 5.48028 | improved_balancer |

## Search Best Candidate

- ID: `cand_000063`
- Name: high_cost_shake
- Status: valid
- Search Gap: 5.110340712098695
- Thought: Multi-start greedy insertion minimizing max distance increase with deterministic tie-breaking. Local search focuses on longest routes using 2-opt, relocate, swap, cross. After no improvement, shakes by removing customers from longest route that have highest contribution to route distance (savings) and reinserting greedily. Repeated cycles improve worst-case gap. Enhanced shake diversification aims to reduce variance and worst-case gap.

## Test Evaluation

- Problem: VRP
- Status: error
- Test gap: inf
- Instances evaluated: 320
- Valid runs: 192 / 320
- Scored runs: 192 / 320
- Partial timeout runs: 0
- Mean gap: 2.888682385860299
- Penalized mean gap: 2.888682385860299
- Mean max route distance: 2.3691593504310378
- Mean total route distance: 12.922401065265504
- Timeout penalty: 0.0
- Median gap: 3.57162811938864
- Worst gap: 13.807488113335873
- Best gap: -56.376156655456974
- Gap by instance size: {'10': None, '100': 5.633480755979188, '20': None, '200': 0.6235191939900785, '50': 2.4090472076116307}
- Gap by truck count: {'1': None, '3': 2.4090472076116307, '5': 5.633480755979188, '9': 0.6235191939900785}
- Error details: error on instance_data_10_000: ValueError: max() iterable argument is empty

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

- History enabled: True
- History size: 20 / 20
- History buckets: 13
- Added candidates: 95
- Duplicate rejections: 0
- History parent selections: 58
- History offspring with history parent: 44
- Final selection from history: False
